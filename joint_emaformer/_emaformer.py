
import torch
import torch.nn as nn
from ._emaformer_layers import Encoder, EncoderLayer
from ._emaformer_layers import FullAttention, AttentionLayer
from ._emaformer_layers import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.cycle_len = configs.cycle
        self.enc_in = configs.enc_in
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model*2),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model*2, configs.d_model*4),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model*4, configs.pred_len),
        )

        self.channel_embedding = nn.Parameter(torch.zeros(configs.enc_in, configs.d_model))
        self.phase_embedding = nn.Embedding(configs.cycle, configs.d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * self.d_model)
        nn.init.xavier_normal_(self.joint_embedding.weight)
        nn.init.xavier_normal_(self.channel_embedding)
      
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase, *, target_index=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        B, L, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)         # covariates (e.g timestamp) can be also embedded as tokens
        
        # Get the four embeddings
        # data_emb = enc_out[:, :N, :]  # Variate token embedding
        channel_emb = self.channel_embedding.expand(enc_out.shape[0], N, -1)  # Channel embedding
        phase_emb = self.phase_embedding(phase.view(-1, 1).expand(B, N))  # Phase embedding
        joint_emb = self.joint_embedding(phase).reshape(B, self.enc_in, self.d_model)  # Joint Channel-Phase embedding
        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_orgin = enc_out

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        channels = slice(None) if target_index is None else slice(target_index, target_index + 1)
        dec_out = self.projector(enc_out[:, channels] + enc_orgin[:, channels]).permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * stdev[:, :, channels] + means[:, :, channels]

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


from types import SimpleNamespace
from torch.utils.data import Dataset


class EMAformer(Model):
    """过去12个月 -> 未来3个月；预测结果为 (batch, 3, 1)。

    核心参数使用官方 run.py 默认值。
    输入必须包含历史目标价格，仅将其编码token送入原预测头。
    forward(x, cycle_index) 的 cycle_index 为预测首月的全局行号 % cycle。
    """

    def __init__(self, input_size, target_index=0, seq_len=12, pred_len=3,
                 d_model=512, n_heads=8, e_layers=2, d_ff=2048,
                 dropout=0.1, output_proj_dropout=0.1, cycle=24,
                 factor=1, activation="gelu", use_norm=True, output_attention=False):
        for name, value in {"input_size": input_size, "seq_len": seq_len,
                            "pred_len": pred_len, "d_model": d_model, "n_heads": n_heads,
                            "e_layers": e_layers, "d_ff": d_ff, "cycle": cycle}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须为正整数")
        if d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if not isinstance(target_index, int) or isinstance(target_index, bool) or not 0 <= target_index < input_size:
            raise ValueError("target_index 必须对应输入中的目标价格列")
        if not 0 <= dropout < 1 or not 0 <= output_proj_dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 范围内")
        if activation not in {"gelu", "relu"}:
            raise ValueError("activation 必须为 gelu 或 relu")
        configs = SimpleNamespace(
            seq_len=seq_len, pred_len=pred_len, enc_in=input_size,
            dec_in=input_size, c_out=input_size, d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, d_ff=d_ff, dropout=dropout,
            output_proj_dropout=output_proj_dropout, cycle=cycle, factor=factor,
            activation=activation, use_norm=use_norm, output_attention=output_attention,
            embed="timeF", freq="m", class_strategy="projection",
        )
        super().__init__(configs)
        self.configs = configs
        self.input_size = input_size
        self.target_index = target_index
        self.feature_names = None

    @classmethod
    def from_dataset(cls, dataset, target="终端非居民天然气价格（分省）", **kwargs):
        names = tuple(dataset.feature_names)
        if target not in names:
            raise ValueError("输入数据集必须包含历史目标价格")
        model = cls(input_size=len(names), target_index=names.index(target),
                    seq_len=dataset.seq_len, pred_len=dataset.pred_len, **kwargs)
        model.feature_names = names
        return model

    def with_phase(self, dataset):
        """给 train/val/test 添加周期位置"""
        if dataset.seq_len != self.seq_len or dataset.pred_len != self.pred_len:
            raise ValueError("模型与数据集的历史/预测长度不一致")
        if len(dataset.feature_names) != self.enc_in:
            raise ValueError("模型与数据集的特征数不一致")
        if self.feature_names is not None and tuple(dataset.feature_names) != self.feature_names:
            raise ValueError("输入特征的名称或顺序与模型创建时不一致")
        return EMAformerDataset(dataset, self.cycle_len)

    def forward(self, x, cycle_index):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x 必须为形状 (batch, {self.seq_len}, {self.enc_in}) 的 Tensor")
        phase = torch.as_tensor(cycle_index, device=x.device)
        if phase.dtype not in (torch.int32, torch.int64):
            raise ValueError("cycle_index 必须为整数")
        if tuple(phase.shape) not in {(x.shape[0],), (x.shape[0], 1)}:
            raise ValueError("cycle_index 形状必须为 (batch,) 或 (batch, 1)")
        phase = phase.reshape(-1).long()
        if ((phase < 0) | (phase >= self.cycle_len)).any():
            raise ValueError(f"cycle_index 必须在 [0, {self.cycle_len}) 范围内")
        prediction, attention = self.forecast(x, None, None, None, phase, target_index=self.target_index)
        if self.output_attention:
            return prediction, attention
        return prediction


class EMAformerDataset(Dataset):
    """输出 (x, y, cycle_index)，周期定义与官方 Dataset_Custom 的 s_end 一致。"""

    def __init__(self, dataset, cycle=24):
        if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 1:
            raise ValueError("cycle 必须为正整数")
        self.dataset = dataset
        self.cycle = cycle

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        # starts 是原始完整序列中预测首月的位置，跨分区及shuffle均保持一致。
        phase = self.dataset.starts[index] % self.cycle
        return x, y, torch.tensor(phase, dtype=torch.long)
