"""项目既有MoFo月度单区域模型，过去6个月预测未来3个月。

核心位于_mofo_core.py，默认d_model=24、head=4、d_layers=1、periodic=6。
通道独立主干后通过线性层融合各变量，输出(batch, 3, 1)。
数据由共用datasets.prepare_data构造，最后历史月相位由预测入口传入。
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset

from ._mofo_core import MoFo as MoFoCore


class MoFo(MoFoCore):
    def __init__(self, input_size, target_index=0, seq_len=6, pred_len=3,
                 d_model=24, head=4, d_layers=1, periodic=6,
                 bias=True, cias=True, channel_fusion=True):
        for name, value in {"input_size": input_size, "seq_len": seq_len,
                            "pred_len": pred_len, "d_model": d_model, "head": head,
                            "d_layers": d_layers, "periodic": periodic}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须为正整数")
        if d_model % head:
            raise ValueError("d_model 必须能被 head 整除")
        if periodic > seq_len:
            raise ValueError("周期分块要求 periodic <= seq_len；6个月历史请用 periodic<=6")
        if target_index is not None and (not isinstance(target_index, int)
                                        or isinstance(target_index, bool)
                                        or not 0 <= target_index < input_size):
            raise ValueError("target_index 必须对应输入中的目标列，或为 None")
        if not channel_fusion and target_index is None:
            raise ValueError("关闭融合时必须输入历史目标价格")
        configs = SimpleNamespace(seq_len=seq_len,
                                  pred_len=pred_len, enc_in=input_size, d_model=d_model,
                                  periodic=periodic, head=head, d_layers=d_layers,
                                  bias=int(bias), cias=int(cias))
        super().__init__(configs)
        self.configs = configs
        self.input_size = input_size
        self.target_index = target_index
        self.feature_names = None
        self.channel_fusion = channel_fusion
        # 官方主干是通道独立的；只有通过输出融合，其他变量才能用于目标预测。
        self.readout = nn.Linear(input_size, 1) if channel_fusion else None
        if self.readout is not None and target_index is not None:
            # 初始结果等于官方目标通道预测，训练后可学习其他变量的贡献。
            with torch.no_grad():
                self.readout.weight.zero_()
                self.readout.weight[0, target_index] = 1
                self.readout.bias.zero_()


    @classmethod
    def from_dataset(cls, dataset, target="终端非居民天然气价格（分省）", **kwargs):
        names = tuple(dataset.feature_names)
        model = cls(input_size=len(names), target_index=names.index(target) if target in names else None,
                    seq_len=dataset.seq_len, pred_len=dataset.pred_len, **kwargs)
        model.feature_names = names
        return model

    def with_phase(self, dataset):
        if dataset.seq_len != self.seq_len or dataset.pred_len != self.pred_len:
            raise ValueError("模型与数据集的窗口长度不一致")
        if len(dataset.feature_names) != self.channels:
            raise ValueError("模型与数据集的特征数不一致")
        if self.feature_names is not None and tuple(dataset.feature_names) != self.feature_names:
            raise ValueError("数据集特征名称或顺序与模型创建时不一致")
        return MoFoDataset(dataset, self.periodic)

    def forward(self, x, phase):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f"x 必须为形状 (batch, {self.seq_len}, {self.channels}) 的 Tensor")
        position = torch.as_tensor(phase, device=x.device)
        if position.dtype not in (torch.int32, torch.int64):
            raise ValueError("phase 必须为整数")
        if tuple(position.shape) not in {(x.shape[0],), (x.shape[0], 1)}:
            raise ValueError("phase 形状必须为 (batch,) 或 (batch, 1)")
        position = position.reshape(-1, 1).long()
        if ((position < 0) | (position >= self.periodic)).any():
            raise ValueError(f"phase 必须在 [0, {self.periodic}) 范围内")
        # 直接传入最后一个历史月的周期位置。
        prediction = super().forecast(x, position)
        if self.readout is not None:
            return self.readout(prediction)
        return prediction[:, :, self.target_index:self.target_index + 1]


class MoFoDataset(Dataset):
    """输出 (x, y, phase)。MoFo使用最后一个历史月的位置，而非首个预测月。"""

    def __init__(self, dataset, periodic=6):
        if not isinstance(periodic, int) or isinstance(periodic, bool) or periodic < 1:
            raise ValueError("periodic 必须为正整数")
        self.dataset = dataset
        self.periodic = periodic

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        phase = (self.dataset.starts[index] - 1) % self.periodic
        return x, y, torch.tensor(phase, dtype=torch.long)
