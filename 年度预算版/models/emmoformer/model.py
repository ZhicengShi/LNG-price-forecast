"""EMMoFormer serial temporal-to-variate network for one region.

The scalar/time encoder uses MoFo-inspired temporal attention. Its gated history
representation is combined with a direct history projection, then an
EMAformer-inspired cross-variate encoder feeds one target-price head. All
normalization statistics come from the supplied history, never future values.
Outputs remain in the global training-fitted standardized target-price units.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ._layers import EncoderBlock, MultiHeadAttention, RMSNorm


class PostLayerEncoderBlock(nn.Module):
    """EMAformer-style attention/GELU block with post LayerNorm."""

    def __init__(self, dim: int, heads: int, dropout: float, ff_multiplier: int):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_multiplier * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_multiplier * dim, dim), nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_norm(x + self.attention_dropout(self.attention(x)))
        return self.ffn_norm(x + self.ffn(x))


class EMMoFormer(nn.Module):
    """[B, seq_len, input_size] history -> [B, pred_len, 1] target price.

    ``starts`` contains each first forecast month's absolute row index relative
    to the first data-table month. For a 2015-01 table, January has phase zero
    and the first six-month window's ``starts`` is 6. ``window_norm`` can be
    ``none``, history-mean ``center``, or reversible history-mean/std ``revin``.
    ``std_floor`` is expressed in the global standardized input units. The
    inverse transformation uses the target variable's own history statistics.
    """

    def __init__(self, input_size: int, target_index: int = 0, seq_len: int = 6,
                 pred_len: int = 3, d_model: int = 64, n_heads: int = 4,
                 temporal_depth: int = 1, channel_depth: int = 1,
                 dropout: float = 0.1, head_dropout: float = 0.1,
                 calendar_cycle: int = 12, first_month_phase: int = 0,
                 window_norm: str = 'revin', std_floor: float = 0.0,
                 temporal_weight_init: float = 0.1,
                 channel_norm: str = 'post_layer', ff_multiplier: int = 4,
                 head_layers: int = 2, head_multiplier: int = 2,
                 head_history: bool = False, embedding_residual: bool = True,
                 use_modulator: bool = True):
        super().__init__()
        integer_fields = dict(input_size=input_size, seq_len=seq_len, pred_len=pred_len,
                              d_model=d_model, n_heads=n_heads,
                              temporal_depth=temporal_depth, channel_depth=channel_depth,
                              calendar_cycle=calendar_cycle, ff_multiplier=ff_multiplier,
                              head_multiplier=head_multiplier)
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if not isinstance(target_index, int) or isinstance(target_index, bool) or not 0 <= target_index < input_size:
            raise ValueError('target_index must identify an input target-price column')
        if not isinstance(first_month_phase, int) or isinstance(first_month_phase, bool) or not 0 <= first_month_phase < calendar_cycle:
            raise ValueError('first_month_phase must lie within the calendar cycle')
        if not isinstance(head_layers, int) or isinstance(head_layers, bool) or head_layers not in (1, 2, 3):
            raise ValueError('head_layers must be 1, 2 or 3')
        for name, value in dict(dropout=dropout, head_dropout=head_dropout).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f'{name} must lie in [0, 1)')
        if not isinstance(std_floor, (int, float)) or isinstance(std_floor, bool) or not math.isfinite(std_floor) or std_floor < 0:
            raise ValueError('std_floor must be a finite nonnegative number')
        if not isinstance(temporal_weight_init, (int, float)) or isinstance(temporal_weight_init, bool) or not math.isfinite(temporal_weight_init) or not 0 < temporal_weight_init < 1:
            raise ValueError('temporal_weight_init must lie strictly between 0 and 1')
        if window_norm not in ('none', 'center', 'revin'):
            raise ValueError('window_norm must be none, center or revin')
        if channel_norm not in ('rms', 'post_layer'):
            raise ValueError('channel_norm must be rms or post_layer')
        if any(not isinstance(flag, bool) for flag in (head_history, embedding_residual, use_modulator)):
            raise ValueError('head_history, embedding_residual and use_modulator must be bool')

        self.input_size, self.target_index = input_size, target_index
        self.seq_len, self.pred_len, self.d_model = seq_len, pred_len, d_model
        self.calendar_cycle, self.first_month_phase = calendar_cycle, first_month_phase
        self.window_norm, self.std_floor = window_norm, float(std_floor)
        self.head_history, self.embedding_residual = head_history, embedding_residual
        self.channel_norm_type = channel_norm
        self.temporal_depth, self.channel_depth = temporal_depth, channel_depth
        self.use_modulator = use_modulator

        self.scalar_embedding = nn.Linear(1, d_model)
        self.relative_age_embedding = nn.Embedding(seq_len, d_model)
        self.calendar_phase_embedding = nn.Embedding(calendar_cycle, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.temporal_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout, calendar_cycle if use_modulator else None)
            for _ in range(temporal_depth)
        ])
        self.temporal_norm = RMSNorm(d_model)
        self.history_aggregation = nn.Linear(seq_len * d_model, d_model)
        self.temporal_weight_logit = nn.Parameter(torch.tensor(math.log(temporal_weight_init / (1 - temporal_weight_init))))
        self.raw_history_projection = nn.Linear(seq_len, d_model)
        self.channel_embedding = nn.Embedding(input_size, d_model)
        self.channel_phase_embedding = nn.Embedding(input_size * calendar_cycle, d_model)
        self.channel_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout) if channel_norm == 'rms' else
            PostLayerEncoderBlock(d_model, n_heads, dropout, ff_multiplier)
            for _ in range(channel_depth)
        ])
        self.channel_norm = RMSNorm(d_model) if channel_norm == 'rms' else nn.LayerNorm(d_model)

        head_input_size = d_model + (seq_len if head_history else 0)
        head_width = head_multiplier * d_model
        head = []
        for _ in range(head_layers - 1):
            head.extend((nn.Linear(head_input_size, head_width), nn.GELU(), nn.Dropout(head_dropout)))
            head_input_size = head_width
        head.append(nn.Linear(head_input_size, pred_len))
        self.prediction_head = nn.Sequential(*head)
        for embedding in (self.relative_age_embedding, self.calendar_phase_embedding,
                          self.channel_embedding, self.channel_phase_embedding):
            nn.init.normal_(embedding.weight, std=0.02)

    @property
    def temporal_weight(self) -> torch.Tensor:
        return self.temporal_weight_logit.sigmoid()

    def calendar_indices(self, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        historical_rows = starts[:, None] - self.seq_len + torch.arange(self.seq_len, device=starts.device)
        return ((historical_rows + self.first_month_phase).remainder(self.calendar_cycle),
                (starts + self.first_month_phase).remainder(self.calendar_cycle))

    def normalize_history(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized history and detached per-variable inverse statistics."""
        if self.window_norm == 'none':
            return x, x.new_zeros((x.shape[0], 1, self.input_size)), x.new_ones((x.shape[0], 1, self.input_size))
        mean = x.mean(dim=1, keepdim=True).detach()
        centered = x - mean
        if self.window_norm == 'center':
            return centered, mean, torch.ones_like(mean)
        scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        scale = scale.clamp_min(self.std_floor)
        return centered / scale, mean, scale

    def forward(self, x: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or tuple(x.shape[1:]) != (self.seq_len, self.input_size):
            raise ValueError(f'x must have shape [batch, {self.seq_len}, {self.input_size}]')
        if x.shape[0] < 1 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('x must be a nonempty finite floating-point tensor')
        starts = torch.as_tensor(starts, device=x.device)
        if starts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('starts must contain integer global first-forecast row indices')
        if tuple(starts.shape) not in {(x.shape[0],), (x.shape[0], 1)}:
            raise ValueError('starts must have shape [batch] or [batch, 1]')
        starts = starts.reshape(-1).long()
        if (starts < self.seq_len).any():
            raise ValueError('starts must leave a full preceding history window')

        batch = x.shape[0]
        normalized, means, scales = self.normalize_history(x)
        raw_history = normalized.transpose(1, 2)
        history_phase, prediction_phase = self.calendar_indices(starts)
        tokens = self.scalar_embedding(raw_history.unsqueeze(-1))
        age = self.relative_age_embedding(torch.arange(self.seq_len, device=x.device))
        tokens = tokens + age[None, None, :, :] + self.calendar_phase_embedding(history_phase)[:, None, :, :]
        tokens = self.embedding_dropout(tokens).reshape(batch * self.input_size, self.seq_len, self.d_model)
        temporal_phases = history_phase.repeat_interleave(self.input_size, dim=0)
        for layer in self.temporal_layers:
            tokens = layer(tokens, temporal_phases)
        temporal_tokens = self.temporal_norm(tokens).reshape(batch, self.input_size, self.seq_len * self.d_model)
        variates = self.raw_history_projection(raw_history) + self.temporal_weight * self.history_aggregation(temporal_tokens)
        channel_ids = torch.arange(self.input_size, device=x.device)
        joint_ids = channel_ids[None, :] * self.calendar_cycle + prediction_phase[:, None]
        variates = variates + self.calendar_phase_embedding(prediction_phase)[:, None, :]
        variates = variates + self.channel_embedding(channel_ids)[None, :, :] + self.channel_phase_embedding(joint_ids)
        variates = self.embedding_dropout(variates)
        embedding = variates
        for layer in self.channel_layers:
            variates = layer(variates)
        variates = self.channel_norm(variates)
        if self.embedding_residual:
            variates = variates + embedding
        target = variates[:, self.target_index, :]
        if self.head_history:
            target = torch.cat((target, raw_history[:, self.target_index, :]), dim=-1)
        output = self.prediction_head(target).reshape(batch, self.pred_len, 1)
        return output * scales[:, :, self.target_index:self.target_index + 1] + means[:, :, self.target_index:self.target_index + 1]
