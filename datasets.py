"""三地区处理后XLSX的共用读取、时间切分、训练历史预处理及窗口。"""
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import re
import numpy as np
import openpyxl
import pandas as pd
import yaml
from provenance import source_info

TARGET = "终端非居民天然气价格（分省）"

DATE = "年月"
GDP = "GDP增速（分省）"

EXCLUDED_FEATURES = {
    "上海": (),
    "浙江": ("LNG进口量（区域）", "天然气产量（区域）"),
    "江苏": ("工业增加值（分省）", "GDP增速（分省）", "非居民天然气消费量（分省）"),
}

FIELDS = [
    'JKM现货价格', 'LNG进口价格', 'TTF期货价格', 'Henry Hub期货价格',
    'Brent原油价格', '管道气进口价格', '美元/人民币汇率',
    'LNG进口量（全国）', '管道气进口量（全国）', 'LNG进口量（区域）',
    '天然气产量（全国总量）', '天然气产量（区域）', 'LNG产量（全国）',
    '门站价基准价格（分省）', '中国液化天然气出厂价格指数',
    '工业增加值（分省）', '制造业PMI增速（全国）', GDP,
    '平均气温（分省）', '采暖度日数HDD（分省）', '非居民天然气消费量（分省）',
    '柴油价格（0#柴油）', '动力煤价格', TARGET,
]

@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values, enabled=True):
        mean = values.mean(axis=0) if enabled else np.zeros(values.shape[1])
        scale = values.std(axis=0) if enabled else np.ones(values.shape[1])
        return cls(mean, np.where(scale == 0, 1.0, scale))

    def transform(self, values):
        return (np.asarray(values) - self.mean) / self.scale

    def inverse_transform(self, values):
        return np.asarray(values) * self.scale + self.mean

class LNGDataset:
    """一个时间分区内的滑动窗口；整个预测区间都属于该分区。"""

    def __init__(self, x, y, dates, feature_names, start, end, seq_len, pred_len):
        self.x, self.y, self.dates = x, y, dates
        self.feature_names = tuple(feature_names)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.starts = range(max(start, seq_len), end - pred_len + 1)
        if not self.starts:
            raise ValueError("分区长度不足以构造窗口，请调整窗口长度或划分比例")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        t = self.starts[index]
        return self.x[t - self.seq_len:t].copy(), self.y[t:t + self.pred_len].copy()

    def sample_months(self, index):
        """返回某个样本的历史月份和预测月份，用于检查或对齐预测结果。"""
        t = self.starts[index]
        return self.dates[t - self.seq_len:t].astype(str), self.dates[t:t + self.pred_len].astype(str)

@dataclass
class DatasetBundle:
    train: LNGDataset
    val: LNGDataset
    test: LNGDataset
    x_scaler: Standardizer
    y_scaler: Standardizer
    dropped_features: tuple

    @property
    def feature_names(self):
        return self.train.feature_names

    def inverse_target(self, predictions):
        """接受 NumPy 预测值，形状可为 (3, 1) 或 (batch, 3, 1)。"""
        return self.y_scaler.inverse_transform(predictions)

class InvalidUnit(ValueError):
    def __init__(self, field):
        super().__init__(f'{field}的单位与模板不一致')
        self.field = field


def read_frame(path, *, expected_units=None, max_months=None):
    """只读取单工作表XLSX：首行字段、第二行单位、年月为YYYY-MM文本。"""
    path = Path(path)
    if path.suffix.lower() != '.xlsx':
        raise ValueError('数据输入只支持XLSX文件')
    with path.open('rb') as stream, closing(openpyxl.load_workbook(stream, read_only=True, data_only=True)) as book:
        if len(book.worksheets) != 1:
            raise ValueError('数据表须仅包含一个工作表')
        sheet = book.worksheets[0]
        values = iter(sheet.values)
        header = list(next(values, ()))
        if header != [DATE, *FIELDS]:
            raise ValueError('处理后数据表须按模板包含年月和24个业务字段')
        units = list(next(values, ()))
        if len(units) != len(header):
            raise ValueError('数据表须提供完整单位行')
        if expected_units is not None:
            for field, actual, expected in zip(header, units, expected_units, strict=True):
                if actual != expected:
                    raise InvalidUnit(field)
        if max_months is not None and sheet.max_row is not None and sheet.max_row > max_months + 2:
            raise ValueError('数据月份数超过服务配置限制')
        rows = list(islice(values, max_months + 1)) if max_months is not None else list(values)
        if max_months is not None and len(rows) > max_months:
            raise ValueError('数据月份数超过服务配置限制')
    while rows and all(value is None for value in rows[-1]):
        rows.pop()
    if not rows:
        raise ValueError('数据表须包含字段行、单位行和月度数据')
    for column in ['LNG进口量（全国）', '管道气进口量（全国）', 'LNG进口量（区域）']:
        if units[header.index(column)] != '吨':
            raise ValueError('只接受已转换为吨的处理后表，不接受原始千克数据')
    if units[header.index(TARGET)] != '元/立方米':
        raise ValueError('目标价格单位须为元/立方米')
    if units[header.index(DATE)] != 'YYYY-MM':
        raise ValueError('第二行年月单位须为YYYY-MM')
    frame = pd.DataFrame(rows, columns=header)
    month_values = frame.pop(DATE)
    if not all(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}', value) for value in month_values):
        raise ValueError('年月须为YYYY-MM文本')
    dates = pd.PeriodIndex(month_values, freq="M")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("月份为空或存在重复")
    frame.index = dates
    if frame.empty or not frame.index.equals(pd.period_range(frame.index[0], frame.index[-1], freq="M")):
        raise ValueError("月份必须按升序连续，不能用跨缺失月份的行构造窗口")
    frame = frame.apply(pd.to_numeric, errors='raise')
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError('数据表含无穷值')
    return frame

def fill_inputs(raw, medians, missing):
    """训练与预测共用缺失处理；中位数由调用方提供，不在此重新拟合。"""
    if missing not in ("ffill", "median", "error"):
        raise ValueError("missing 只能为 ffill、median 或 error")
    if np.isinf(raw.to_numpy(dtype=float)).any():
        raise ValueError("输入含无穷值")
    if missing == "error" and raw.isna().any().any():
        raise ValueError("输入存在缺失，请选择填充策略或其他变量")
    values = (raw.ffill() if missing == "ffill" else raw).fillna(pd.Series(medians)).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("预处理后仍存在无效输入")
    return values


REGIONS = {'shanghai': '上海', 'zhejiang': '浙江', 'jiangsu': '江苏'}


def resolve_path(value, config_path):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(config_path).resolve().parent/path).resolve()


def load_global(config_path):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    if set(cfg['data']) != set(REGIONS):
        raise ValueError('公共配置data须指定shanghai、zhejiang、jiangsu三个地区')
    cfg['data'] = {r: resolve_path(value, config_path) for r, value in cfg['data'].items()}
    return cfg


def selected_names(frame, cfg, region, profile, fit_end):
    selection = cfg['features'][profile]
    if profile == 'selected':
        selection = selection[region]
    elif isinstance(selection, dict):
        selection = selection[region]
    all_features = isinstance(selection, str) and selection == 'all'
    if not all_features and not isinstance(selection, list):
        raise ValueError('变量配置须为all或完整列名列表')
    excluded = EXCLUDED_FEATURES[REGIONS[region]]
    names = [name for name in frame if name not in excluded] if all_features else list(selection)
    if not names or len(names) != len(set(names)) or any(c not in frame or c in excluded for c in names):
        raise ValueError('所选变量为空、重复、不存在或已停用')
    if TARGET not in names:
        names.insert(0, TARGET)
    dropped = tuple(c for c in names if frame[c].iloc[:fit_end].isna().all())
    if dropped and not all_features:
        raise ValueError(f'自选变量在训练区间全空：{dropped}')
    names = [c for c in names if c not in dropped]
    return names, dropped


def prepare_data(cfg, region, profile):
    """固定时间边界；预处理拟合仅使用训练期，与原单区域实验一致。"""
    if region not in REGIONS or profile not in ('all', 'selected'):
        raise ValueError('地区或变量方案无效')
    path = cfg['data'][region]
    full = read_frame(path)
    settings = cfg['period']
    for key in ['start', 'train_end', 'validation_end', 'test_end']:
        if not re.fullmatch(r'\d{4}-\d{2}', str(settings[key])):
            raise ValueError(f'period.{key}须使用YYYY-MM')
    start, train_end, val_end, test_end = [pd.Period(settings[k], freq='M') for k in
                                         ['start', 'train_end', 'validation_end', 'test_end']]
    if not start <= train_end < val_end < test_end:
        raise ValueError('训练、验证、测试月份须按时间顺序排列')
    months = pd.period_range(start, test_end, freq='M')
    if not months.isin(full.index).all():
        raise ValueError('数据表未覆盖配置中的完整实验区间')
    frame = full.loc[months].copy()
    fit_end = frame.index.get_loc(train_end)+1
    validation_end = frame.index.get_loc(val_end)+1
    seq, horizon = cfg['window']['seq_len'], cfg['window']['pred_len']
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (seq, horizon)) or horizon != 3:
        raise ValueError('历史窗口须为正整数，当前实验输出固定为3个月')
    if fit_end < seq+horizon or min(validation_end-fit_end, len(frame)-validation_end) < horizon:
        raise ValueError('时间分区不足以构造训练或评价窗口')
    labels = frame[TARGET].to_numpy(dtype=float)[:, None]
    if not np.isfinite(labels).all() or (labels <= 0).any():
        raise ValueError('真实目标价格须为正且不可缺失，不允许填充标签')
    names, dropped = selected_names(frame, cfg, region, profile, fit_end)
    raw = frame[names]
    missing = cfg['preprocessing']['missing']
    medians = raw.iloc[:fit_end].median()
    values = fill_inputs(raw, medians, missing)
    x_scaler = Standardizer.fit(values[:fit_end])
    y_scaler = Standardizer.fit(labels[:fit_end])
    x = x_scaler.transform(values).astype(np.float32)
    y = y_scaler.transform(labels).astype(np.float32)
    bounds = dict(train=[0, fit_end], val=[fit_end, validation_end], test=[validation_end, len(frame)])
    partitions = [LNGDataset(x, y, frame.index, names, a, b, seq, horizon) for a, b in bounds.values()]
    data = DatasetBundle(*partitions, x_scaler, y_scaler, dropped)
    metadata = dict(region=region, feature_mode=profile, features=names, dropped_features=list(dropped),
        missing=missing, medians=medians.to_dict(),
        x_scaler=dict(mean=x_scaler.mean.tolist(), scale=x_scaler.scale.tolist()),
        y_scaler=dict(mean=y_scaler.mean.tolist(), scale=y_scaler.scale.tolist()),
        first_month=str(frame.index[0]), bounds=bounds, training_cutoff_month=str(train_end),
        split_end_months=dict(val=str(val_end), test=str(test_end)), data_source=source_info(path))
    return data, frame, metadata
