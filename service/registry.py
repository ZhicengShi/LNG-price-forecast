"""读取已登记权重，核对评价证据、训练协议和输入布局。"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from datasets import read_frame, resolve_path
from CoRegionLng._runtime import load_checkpoint
from CoRegionLng.analysis import as_arrays
from CoRegionLng.data import REGIONS, TARGET
from .storage import now


@dataclass
class RegisteredModel:
    model: object
    saved: dict
    version: str
    registered_at: str
    loaded_at: str
    fixed_test: pd.DataFrame | None
    backtest: pd.DataFrame | None
    fixed_test_version: str | None
    backtest_version: str | None


def checked_table(spec, registry_path, profile, references):
    path = resolve_path(spec['predictions'], registry_path)
    table = pd.read_csv(path)
    if set(table.profile) != {profile}:
        raise ValueError('评价证据的变量方案不匹配')
    as_arrays(table)
    first = references[REGIONS[0]].index[0]
    for row in table.itertuples():
        month = pd.Period(row.target_month, freq='M')
        if (month != first+int(row.origin_index)+int(row.horizon)-1
                or pd.Period(row.history_cutoff, freq='M') != first+int(row.origin_index)-1
                or pd.Period(row.first_forecast_month, freq='M') != first+int(row.origin_index)):
            raise ValueError('评价证据的月份、起点与步长不一致')
        actual = references[row.region].at[month, TARGET]
        if not np.isclose(actual, row.actual, rtol=0, atol=1e-12):
            raise ValueError('评价证据的真实值与参考数据不一致')
    return table, spec['version']


def load_registry(path, device, store):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if config['registry_version'] != 2 or set(config['models']) != {'all', 'selected'}:
        raise ValueError('模型登记须包含all和selected两个方案')
    reference_paths = {r: resolve_path(config['reference_data'][r], path) for r in REGIONS}
    references = {r: read_frame(p) for r, p in reference_paths.items()}
    original = json.loads(resolve_path(config['reference_weights_index'], path).read_text(encoding='utf-8'))
    layouts = {item['profile']: item['input_layout'] for item in original}
    result = {}
    for profile, spec in config['models'].items():
        checkpoint = resolve_path(spec['checkpoint'], path)
        version = spec['model_version']
        if not isinstance(version,str) or not version.startswith(profile+'-') or len(version)>80:
            raise ValueError('模型版本须以变量方案开头，且不超过80字符')
        model, saved = load_checkpoint(checkpoint, device)
        if saved['profile'] != profile:
            raise ValueError('模型登记的变量方案与权重不匹配')
        if saved.get('model_version',version) != version:
            raise ValueError('登记的模型版本与权重记录不一致，请使用配套版本编号')
        fixed = backtest = fixed_version = backtest_version = None
        if spec.get('fixed_test'):
            if spec['fixed_test']['model_version'] != version:
                raise ValueError('固定权重测试证据属于其他权重；请移除旧证据或登记新证据')
            fixed, fixed_version = checked_table(spec['fixed_test'], path, profile, references)
            if (fixed.history_cutoff < saved['training_cutoff_month']).any():
                raise ValueError('固定权重测试范围早于训练截止月')
        if spec.get('backtest'):
            evidence = spec['backtest']
            protocol = json.loads(resolve_path(evidence['protocol'], path).read_text(encoding='utf-8'))
            model_config = json.loads(resolve_path(evidence['model_config'], path).read_text(encoding='utf-8'))
            if (protocol['first_month'] != saved['layout']['first_month']
                    or protocol['selection_cutoff_month'] != saved['selection_cutoff_month']
                    or saved['layout'] != layouts[profile]
                    or saved['parameters'] != model_config['parameters']
                    or saved['epochs'] != model_config['training']['epochs']
                    or saved['lr_schedule'] != model_config['training']['lr_schedule']):
                raise ValueError('回测校准协议与当前模型的结构、变量布局或训练计划不一致')
            backtest, backtest_version = checked_table(evidence, path, profile, references)
        registered_at = store.register_model(version, dict(profile=profile, checkpoint=str(checkpoint)))
        result[profile] = RegisteredModel(model, saved, version, registered_at, now(),
                                          fixed, backtest, fixed_version, backtest_version)
    return result, references, reference_paths
