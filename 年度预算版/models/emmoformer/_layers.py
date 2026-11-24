"""Attention, normalization and feed-forward layers used by EMMoFormer."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return x * inverse_rms.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        # Two-times expansion limits parameters for the short monthly sample.
        self.value = nn.Linear(dim, 2 * dim)
        self.gate = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(F.silu(self.gate(x)) * self.value(x)))


class MultiHeadAttention(nn.Module):
    """Ordinary self attention with an optional additive attention-logit prior."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        logits = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        if bias is not None:
            logits = logits + bias
        attention = self.dropout(logits.softmax(-1))
        result = (attention @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.output(result)


class CircularDistanceModulator(nn.Module):
    """A weak, learnable MoFo-inspired prior; each head has its own parameters.

    For circular month distance d and P calendar positions,
      f(d) = sigmoid(a * (b - d)) + exp(-d) * sigmoid(-a * b),
      a = sigmoid(a_raw), b = P * sigmoid(b_raw).
    f is positive and f(0)=1. Attention logits receive strength * log(f).
    Learning separate bounded strengths per head is a change from original MoFo.
    """

    def __init__(self, heads: int, cycle: int):
        super().__init__()
        self.cycle = cycle
        self.a_raw = nn.Parameter(torch.zeros(heads))
        self.b_raw = nn.Parameter(torch.zeros(heads))
        self.strength_raw = nn.Parameter(torch.full((heads,), -3.0))

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        delta = (phases[:, :, None] - phases[:, None, :]).abs()
        distance = torch.minimum(delta, self.cycle - delta).to(self.a_raw.dtype)
        distance = distance[:, None, :, :]
        a = self.a_raw.sigmoid()[None, :, None, None]
        b = (self.cycle * self.b_raw.sigmoid())[None, :, None, None]
        strength = self.strength_raw.sigmoid()[None, :, None, None]
        positive = torch.sigmoid(a * (b - distance)) + torch.exp(-distance) * torch.sigmoid(-a * b)
        return strength * positive.clamp_min(1e-8).log()


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, calendar_cycle: int | None = None):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(dim, heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.modulator = CircularDistanceModulator(heads, calendar_cycle) if calendar_cycle else None

    def forward(self, x: torch.Tensor, phases: torch.Tensor | None = None) -> torch.Tensor:
        bias = self.modulator(phases) if self.modulator is not None else None
        x = x + self.residual_dropout(self.attention(self.attention_norm(x), bias))
        return x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
