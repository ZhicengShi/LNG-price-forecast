"""LSTM多变量输入、单变量三步预测。

数据由共用datasets.prepare_data构造，输入(batch, seq_len, input_size)，
输出(batch, 3, 1)。使用最后隐藏状态直接预测，不需要未来解释变量。
输出采用训练目标的标准化尺度，由预测入口按保存的参数恢复价格。
"""
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1,
                 dropout=0.1, seq_len=6, pred_len=3):
        super().__init__()
        for name, value in {"input_size": input_size, "hidden_size": hidden_size,
                            "num_layers": num_layers, "seq_len": seq_len,
                            "pred_len": pred_len}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须为正整数")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 范围内")
        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 单层LSTM也通过输出层dropout进行正则化；回归输出不加激活函数。
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, pred_len))

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """从 data.train 自动取得筛选后的变量数、历史长度和预测长度。"""
        return cls(input_size=len(dataset.feature_names), seq_len=dataset.seq_len,
                   pred_len=dataset.pred_len, **kwargs)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x 必须为 torch.Tensor，可通过 DataLoader 获取")
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.input_size):
            raise ValueError(f"输入应为 (batch, {self.seq_len}, {self.input_size})，实际为 {tuple(x.shape)}")
        # 每个窗口独立初始化隐藏状态，不在批次之间传递历史。
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).unsqueeze(-1)
