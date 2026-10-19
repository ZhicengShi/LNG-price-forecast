"""配置、数据版本、严格权重读取与统一结果表。"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml

from datasets import load_global, read_frame, resolve_path
from metrics import metric_row, report
from .data import REGIONS, NAMES, build_frame
from .model import JointEMAformer

HERE = Path(__file__).resolve().parent
MODEL_NAME = '三地区融合EMAformer'
FORMAT = 'joint_emaformer_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def replace(temporary, target):
    for attempt in range(10):
        try:
            Path(temporary).replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.05*2**attempt, .5))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    replace(temporary, path)


def csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding='utf-8-sig')
    replace(temporary, path)


def save_checkpoint(path, saved):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(saved, temporary)
    replace(temporary, path)


def device_for(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA不可用，请在公共YAML设置runtime.device为cpu或auto')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('当前复现仅支持cpu或cuda')
    return device


def configure(shared, override=None):
    runtime = shared['runtime']
    threads = runtime['threads']
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError('runtime.threads须为正整数')
    torch.set_num_threads(threads)
    return device_for(override or runtime['device'])


def run_directory(parent, prefix):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stem = prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M')
    for i in range(1, 100000):
        path = parent/(stem if i == 1 else f'{stem}_{i}')
        try:
            path.mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError('无法建立独立运行目录')


def common_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--data-config', type=Path)
    parser.add_argument('--profile', nargs='+', choices=['all', 'selected'])
    parser.add_argument('--device')
    parser.add_argument('--output-dir', type=Path)
    return parser


def config_paths(cli):
    if cli.config:
        if cli.profile:
            raise ValueError('指定config后无需另选profile，变量方案由配置确定')
        return [cli.config.resolve()]
    return [HERE/'configs'/f'{p}.yaml' for p in (cli.profile or ['all', 'selected'])]


def read_config(path, data_config=None):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if cfg['model'] != 'JointEMAformer' or cfg['profile'] not in ('all', 'selected'):
        raise ValueError('配置须为JointEMAformer及all或selected变量方案')
    shared_path = Path(data_config).resolve() if data_config else resolve_path(cfg['data_config'], path)
    shared = load_global(shared_path)
    if not isinstance(cfg['training']['epochs'], int) or isinstance(cfg['training']['epochs'], bool) or cfg['training']['epochs'] < 1:
        raise ValueError('training.epochs须为正整数')
    schedule = cfg['training']['lr_schedule']
    if schedule is None:
        schedule = [cfg['parameters']['lr']]*cfg['training']['epochs']
        cfg['training']['lr_schedule'] = schedule
    if (not isinstance(schedule, list) or len(schedule) != cfg['training']['epochs'] or not schedule
            or any(not isinstance(v, (float, int)) or not np.isfinite(v) or v <= 0 for v in schedule)):
        raise ValueError('lr_schedule须包含与epochs数量一致的正学习率')
    if cfg['training']['seed'] < 0:
        raise ValueError('训练种子须为非负整数')
    parameters = cfg['parameters']
    validate_parameters(parameters)
    if cfg['prediction']['mode'] not in ('test', 'future') or shared['window']['pred_len'] != 3:
        raise ValueError('预测模式须为test或future，当前输出固定未来3个月')
    return cfg, shared, path, shared_path


def validate_parameters(p):
    for key in ('seq_len', 'd_model', 'n_heads', 'e_layers', 'ff_multiplier', 'head_layers',
                'head_multiplier', 'batch_size', 'max_epochs', 'patience'):
        if not isinstance(p[key], int) or isinstance(p[key], bool) or p[key] < 1:
            raise ValueError(f'{key}须为正整数')
    if p['d_model'] % p['n_heads'] or not isinstance(p['cycle'], int) or p['cycle'] < 0:
        raise ValueError('d_model须整除n_heads，cycle须为非负整数')
    for key in ('dropout', 'embedding_dropout', 'attention_dropout', 'head_dropout'):
        if not 0 <= p[key] < 1:
            raise ValueError(f'{key}须在[0,1)内')
    for key, choices in dict(output_mode=('level', 'delta'), missing=('linear_history', 'ffill'),
                             loss=('mse', 'smooth_l1', 'mae', 'mape'), optimizer=('Adam', 'AdamW'),
                             scheduler=('none', 'cosine', 'plateau'), activation=('gelu', 'relu')).items():
        if p[key] not in choices:
            raise ValueError(f'{key}不在支持的选项内')
    if not np.isfinite(p['lr']) or p['lr'] <= 0 or min(p['residual_scale'], p['weight_decay'], p['grad_clip']) < 0:
        raise ValueError('学习率须为正数，修正量系数、权重衰减和梯度裁剪须非负')


def read_sources(shared, end=None):
    frames, versions = {}, {}
    for region in REGIONS:
        path = shared['data'][region]
        versions[region] = digest(path)
        frame = read_frame(path)
        first = pd.Period(shared['period']['start'], freq='M')
        last = pd.Period(end, freq='M') if end is not None else frame.index[-1]
        months = pd.period_range(first, last, freq='M')
        if not len(months) or not months.isin(frame.index).all():
            raise ValueError('数据须覆盖配置的完整月份')
        frames[region] = frame.loc[months].copy()
        if digest(path) != versions[region]:
            raise ValueError('读取期间源数据发生变化')
    if any(not frames[REGIONS[0]].index.equals(frames[r].index) for r in REGIONS):
        raise ValueError('三地区可用历史月份须严格一致')
    return frames, versions


def assert_versions(shared, versions):
    if any(digest(shared['data'][r]) != versions[r] for r in REGIONS):
        raise ValueError('运行期间源数据发生变化')


def frame_for(shared, profile, end=None, layout=None):
    frames, versions = read_sources(shared, end)
    first = frames[REGIONS[0]]
    train_end = int((first.index <= pd.Period(shared['period']['train_end'], freq='M')).sum())
    frame, layout = build_frame(frames, profile, shared['features']['selected'], train_end,
                                shared['features']['all'], saved_layout=layout)
    return frame, layout, versions


def cutoff_index(frame, month):
    if month == 'latest':
        return len(frame)
    month = pd.Period(str(month), freq='M')
    if month not in frame.index:
        raise ValueError('训练或评价截止月份超出可用数据')
    return int(frame.index.get_loc(month))+1


def package_checkpoint(base, profile, frame, versions, selection_end, audit=None):
    saved = deepcopy(base)
    saved.update(format_version=FORMAT, architecture='three_region_emaformer_v1', profile=profile,
                 data_sha256=versions, training_cutoff_month=str(frame.index[saved['fit_end']-1]),
                 selection_cutoff_month=str(selection_end), price_unit='元/立方米')
    if audit is not None:
        saved['training_audit'] = audit
    return saved


def load_checkpoint(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format_version') != FORMAT or saved['layout']['regions'] != list(REGIONS):
        raise ValueError('权重格式无效，须使用本目录的三地区配套权重')
    model = JointEMAformer(saved['layout'], saved['parameters']).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model, saved


def prediction_rows(frame, starts, actual, forecast, profile, cutoff=None):
    rows = []
    for j, t in enumerate(starts):
        for h in range(3):
            for i, region in enumerate(REGIONS):
                row = dict(profile=profile, region=region, origin_index=int(t),
                           history_cutoff=str(frame.index[t-1]), first_forecast_month=str(frame.index[t]),
                           target_month=str(frame.index[t+h]), horizon=h+1,
                           actual=float(actual[j, h, i]), prediction=float(forecast[j, h, i]))
                if cutoff is not None:
                    row.update(model_training_cutoff=str(cutoff), latest_observed_month=str(frame.index[t-1]))
                rows.append(row)
    return rows


def metrics_rows(table):
    table = pd.DataFrame(table)
    rows = []
    for (profile, region), group in table.groupby(['profile', 'region'], sort=False):
        ordered = group.sort_values(['origin_index', 'horizon'])
        counts = ordered.groupby('origin_index').horizon.apply(list)
        if not all(values == [1, 2, 3] for values in counts):
            raise ValueError('指标表存在缺失或重复预测步')
        actual = ordered.actual.to_numpy().reshape(-1, 3, 1)
        forecast = ordered.prediction.to_numpy().reshape(-1, 3, 1)
        rows.append(dict(地区=NAMES[region], 变量方案=profile, 起点数=len(actual),
                         **metric_row(MODEL_NAME, report(actual, forecast))))
    return rows


def save_prediction_results(output, rows, prefix='test'):
    csv(Path(output)/f'{prefix}_predictions.csv', rows)
    metrics = metrics_rows(rows)
    csv(Path(output)/f'{prefix}_metrics.csv', metrics)
    save(Path(output)/f'{prefix}_metrics.json', metrics)
