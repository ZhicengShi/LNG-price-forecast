"""三地区输入布局、历史预处理和窗口；共用datasets.py读取处理后表。"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from datasets import TARGET, EXCLUDED_FEATURES, read_frame, selected_names

COMMON = ['JKM现货价格', 'LNG进口价格', 'TTF期货价格', 'Henry Hub期货价格', 'Brent原油价格', '管道气进口价格', '美元/人民币汇率', 'LNG进口量（全国）', '管道气进口量（全国）', '天然气产量（全国总量）', 'LNG产量（全国）', '中国液化天然气出厂价格指数', '制造业PMI增速（全国）', '柴油价格（0#柴油）', '动力煤价格']

REGIONS = ('shanghai', 'zhejiang', 'jiangsu')
NAMES = dict(shanghai='上海', zhejiang='浙江', jiangsu='江苏')


def build_frame(sources, profile, selected, train_end=95, all_features='all', saved_layout=None):
    frames = {r: sources[r].copy() if isinstance(sources[r], pd.DataFrame) else read_frame(sources[r]) for r in REGIONS}
    first = frames[REGIONS[0]]
    for r in REGIONS:
        if not first.index.equals(frames[r].index):
            raise ValueError('三地区年月须严格一致')
        np.testing.assert_allclose(first[COMMON], frames[r][COMMON], rtol=1e-12, atol=1e-12, equal_nan=True)
        labels = frames[r][TARGET].to_numpy(dtype=float)
        if not np.isfinite(labels).all() or (labels <= 0).any():
            raise ValueError('目标价格须为正且不可缺失，不允许插补标签')
    if saved_layout is not None:
        common = saved_layout['input_schema']['common']
        local = {r: saved_layout['input_schema'][r] for r in REGIONS}
    elif profile == 'all':
        config = {'features': {'all': all_features}}
        names = {r: selected_names(frames[r], config, r, 'all', train_end)[0] for r in REGIONS}
        common = [c for c in COMMON if all(c in names[r] for r in REGIONS)]
        local = {r: [TARGET]+[c for c in names[r] if c not in common and c != TARGET] for r in REGIONS}
    elif profile == 'selected':
        common = []
        config = {'features': {'selected': selected}}
        local = {r: selected_names(frames[r], config, r, 'selected', train_end)[0] for r in REGIONS}
        local = {r: [TARGET]+[c for c in local[r] if c != TARGET] for r in REGIONS}
    else:
        raise ValueError('变量方案须为all或selected')
    frame = first[common].copy()
    for r in REGIONS:
        if any(c in EXCLUDED_FEATURES[NAMES[r]] or c not in frames[r] for c in local[r]):
            raise ValueError('所选变量不存在或已停用')
        for c in local[r]:
            frame[r+'::'+c] = frames[r][c]
    features = list(frame)
    shared = list(range(len(common)))
    layout = dict(input_size=len(features), pred_len=3, first_month=str(frame.index[0]),
                  features=features, regions=list(REGIONS),
                  target_indices=[features.index(r+'::'+TARGET) for r in REGIONS],
                  branch_indices=[shared+[features.index(r+'::'+c) for c in local[r]] for r in REGIONS],
                  input_schema=dict(common=common, **local))
    if saved_layout is not None and layout != saved_layout:
        raise ValueError('数据列布局与权重不一致')
    return frame, layout


@dataclass
class Windows:
    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    actual: np.ndarray
    starts: list


def windows(frame, layout, preprocess, start, end, seq_len):
    horizon = layout['pred_len']
    starts = list(range(max(seq_len, start), end-horizon+1))
    if not starts:
        raise ValueError('历史或评分区间太短')
    x = []
    for t in starts:
        # 每个窗口只插补截至t-1的历史，不能跨预测起点读取未来变量。
        history = fill_history(frame.iloc[:t], preprocess['medians'], preprocess['method']).iloc[-seq_len:]
        values = (history.to_numpy(dtype=float)-np.asarray(preprocess['x_mean']))/np.asarray(preprocess['x_scale'])
        x.append(values.astype(np.float32))
    labels = frame.iloc[:, layout['target_indices']].to_numpy(dtype=float)
    actual = np.stack([labels[t:t+horizon] for t in starts])
    y = ((actual-np.asarray(preprocess['y_mean']))/np.asarray(preprocess['y_scale'])).astype(np.float32)
    x = np.stack(x)
    if not np.isfinite(x).all() or not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError('输入无效或目标缺失，不允许插补标签')
    return Windows(torch.from_numpy(x), torch.from_numpy(y), torch.tensor(starts), actual, starts)


class PreparedData:
    def __init__(self, frame, layout, device):
        self.frame, self.layout, self.device = frame, layout, device
        self.cache = {}

    def partition(self, p, fit_end, start, end):
        key = (p['missing'], p['seq_len'], fit_end, start, end)
        if key not in self.cache:
            prep = fit_preprocessing(self.frame, self.layout, fit_end, p['missing'])
            self.cache[key] = (windows(self.frame, self.layout, prep, start, end, p['seq_len']), prep)
        return self.cache[key]


def fit_preprocessing(frame,layout,fit_end,method):
    medians = frame.iloc[:fit_end].median()
    if medians.isna().any():
        raise ValueError('当前训练历史存在全空列')
    fitted = fill_history(frame.iloc[:fit_end],medians,method).to_numpy(dtype=float)
    mean,std = fitted.mean(axis=0),fitted.std(axis=0)
    scale = np.where(std==0,1.,std)
    labels = frame.iloc[:fit_end,layout['target_indices']].to_numpy(dtype=float)
    y_mean,y_std = labels.mean(axis=0),labels.std(axis=0)
    y_scale = np.where(y_std==0,1.,y_std)
    np.testing.assert_allclose(mean[layout['target_indices']],y_mean,atol=1e-12)
    np.testing.assert_allclose(scale[layout['target_indices']],y_scale,atol=1e-12)
    return dict(medians=medians.to_dict(),x_mean=mean.tolist(),x_scale=scale.tolist(),
                y_mean=y_mean.tolist(),y_scale=y_scale.tolist(),fit_end=fit_end,method=method)


def fill_history(history,medians,method):
    if method=='linear_history':
        history = history.interpolate(method='linear',limit_area='inside')
    elif method!='ffill':
        raise ValueError('未知缺失处理')
    return history.ffill().fillna(pd.Series(medians))

