"""单区域模型共用的配置、权重、预处理和预测输出；仅依赖本目录。"""
import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from datasets import LNGDataset, TARGET, fill_inputs, read_frame, resolve_path, load_global
from metrics import report, metric_row, save_table
from provenance import source_info, without_file_checks

ROOT = Path(__file__).resolve().parent
REGIONS = ('shanghai', 'zhejiang', 'jiangsu')
NAMES = dict(shanghai='上海', zhejiang='浙江', jiangsu='江苏')
MODEL_FOLDERS = dict(LSTM='lstm', EMAformer='emaformer', MoFo='mofo', EMMoFormer='emmoformer')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def device_for(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA不可用，请在YAML中指定cpu')
    return device


def get_model_class(name):
    return getattr(importlib.import_module(MODEL_FOLDERS[name]+'.model'), name)


def load_settings(model_name, argv=None, prediction=False):
    folder = ROOT/MODEL_FOLDERS[model_name]
    parser = argparse.ArgumentParser(description=f'{model_name}单区域'+('预测' if prediction else '训练'))
    parser.add_argument('--config', type=Path, default=folder/'config.yaml')
    parser.add_argument('--data-config', type=Path, help='覆盖公共config.yaml路径')
    parser.add_argument('--region', nargs='+', choices=REGIONS)
    parser.add_argument('--profile', nargs='+', choices=['all', 'selected'])
    parser.add_argument('--device', help='覆盖YAML中的运行设备')
    parser.add_argument('--output-dir', type=Path)
    if prediction:
        parser.add_argument('--mode', choices=['test', 'val', 'future'])
        parser.add_argument('--checkpoint', type=Path, help='指定一份权重；地区与变量方案从权重中读取')
    else:
        parser.add_argument('--epochs', type=int, help='覆盖最大训练轮数')
    args = parser.parse_args(argv)
    path = args.config.resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if cfg['model'] != model_name:
        raise ValueError('模型配置与入口不匹配')
    global_path = args.data_config.resolve() if args.data_config else resolve_path(cfg['data_config'], path)
    global_cfg = load_global(global_path)
    run = cfg['prediction' if prediction else 'training'].copy()
    if args.region:
        run['regions'] = args.region
    if args.profile:
        run['profiles'] = args.profile
    for name, valid in [('regions', REGIONS), ('profiles', ('all', 'selected'))]:
        if not run[name] or len(run[name]) != len(set(run[name])) or any(x not in valid for x in run[name]):
            raise ValueError(f'{name}须非空、不重复且取值有效')
    if prediction and args.mode:
        run['mode'] = args.mode
    if not prediction and args.epochs is not None:
        run['epochs'] = args.epochs
    runtime = global_cfg['runtime'].copy()
    if args.device:
        runtime['device'] = args.device
    if not isinstance(runtime['threads'], int) or runtime['threads'] < 1:
        raise ValueError('threads须为正整数')
    torch.set_num_threads(runtime['threads'])
    run['output_dir'] = args.output_dir.resolve() if args.output_dir else resolve_path(run['output_dir'], path)
    if prediction:
        run['checkpoint_dir'] = resolve_path(run['checkpoint_dir'], path)
        if run['mode'] not in ('test', 'val', 'future'):
            raise ValueError('预测mode须为test、val或future')
        if args.checkpoint:
            if args.region or args.profile:
                raise ValueError('指定checkpoint时无需另选region/profile，按权重中的记录运行')
            run['checkpoint'] = args.checkpoint.resolve()
    return cfg, global_cfg, runtime, run, path


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
            continue
    raise RuntimeError('无法创建独立输出目录')


def forward(model, x, starts, name):
    if name == 'EMAformer':
        value = model(x, starts % model.cycle_len)
    elif name == 'EMMoFormer':
        value = model(x, starts)
    elif name == 'MoFo':
        value = model(x, (starts-1) % model.periodic)
    else:
        value = model(x)
    return value[0] if isinstance(value, tuple) else value


def evaluate(model, dataset, scaler, device, name, batch_size=16):
    """保持原实验DataLoader、float32标签及逆标准化口径。"""
    model.eval()
    predictions, actuals, offset = [], [], 0
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            starts = torch.tensor(list(dataset.starts[offset:offset+len(x)]), device=device)
            predictions.append(forward(model, x.to(device), starts, name).cpu().numpy())
            actuals.append(y.numpy())
            offset += len(x)
    pred = np.concatenate(predictions)*np.asarray(scaler['scale'])+np.asarray(scaler['mean'])
    actual = np.concatenate(actuals)*np.asarray(scaler['scale'])+np.asarray(scaler['mean'])
    return report(actual, pred), actual, pred


def save_evaluation(folder, model_name, config, dataset, result, actual, predicted, prefix='test'):
    folder = Path(folder)
    rows = []
    for i, t in enumerate(dataset.starts):
        for h in range(dataset.pred_len):
            a, p = float(actual[i, h, 0]), float(predicted[i, h, 0])
            rows.append(dict(地区=NAMES[config['region']], 变量方案=config['feature_mode'], 模型=model_name,
                历史截止月份=str(dataset.dates[t-1]), 预测月份=str(dataset.dates[t+h]), 预测步=h+1,
                真实价格=a, 预测价格=p, APE_percent=abs((p-a)/a)*100))
    pd.DataFrame(rows).to_csv(folder/f'{prefix}_predictions.csv', index=False, encoding='utf-8-sig')
    save_json(folder/f'{prefix}_metrics.json', result)
    save_table(folder/f'{prefix}_metrics.csv', [metric_row(model_name, result)])


def load_checkpoint(path, model_name, device):
    saved = without_file_checks(torch.load(path, map_location='cpu', weights_only=True))
    if saved.get('format_version') != 'single_region_v1' or saved['config']['model'] != model_name:
        raise ValueError('权重须属于本复现目录及当前模型')
    cfg = saved['config']
    model = get_model_class(model_name)(**cfg['model_kwargs']).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.eval(), saved


def load_inputs(frame, cfg):
    if str(frame.index[0]) != cfg['first_month']:
        raise ValueError('请提供从权重首月开始的完整历史，以保持填充及周期位置一致')
    if len(frame) < cfg['bounds']['train'][1]:
        raise ValueError('输入历史早于权重训练截止月')
    target = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError('历史目标价格不可缺失或非正，不填充标签')
    raw = frame[cfg['features']]
    value = fill_inputs(raw, cfg['medians'], cfg['missing'])
    scaler = cfg['x_scaler']
    x = ((value-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('预处理后存在无效输入')
    return x


def predict_one(checkpoint_path, global_cfg, model_name, device, output, mode='test', batch_size=16):
    model, saved = load_checkpoint(checkpoint_path, model_name, device)
    cfg = saved['config']
    source = global_cfg['data'][cfg['region']]
    full = read_frame(source)
    frame = full.loc[full.index >= pd.Period(cfg['first_month'], freq='M')]
    # 评价仍按权重保存的时间边界进行；未来预测沿用保存的预处理。
    x = load_inputs(frame, cfg)
    seq, horizon = cfg['model_kwargs']['seq_len'], cfg['model_kwargs']['pred_len']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if mode == 'future':
        if len(frame) < max(seq, cfg['bounds']['val'][1]):
            raise ValueError('历史尚未覆盖该权重的验证选模截止月')
        with torch.inference_mode():
            value = forward(model, torch.from_numpy(x[-seq:])[None].to(device),
                            torch.tensor([len(x)], device=device), model_name).cpu().numpy()[0]
        prices = value*np.asarray(cfg['y_scaler']['scale'])+np.asarray(cfg['y_scaler']['mean'])
        if not np.isfinite(prices).all():
            raise ValueError('预测值非有限数值')
        dates = pd.period_range(frame.index[-1]+1, periods=horizon, freq='M')
        table = pd.DataFrame([dict(地区=NAMES[cfg['region']], 变量方案=cfg['feature_mode'], 模型=model_name,
            预测月份=str(month), 预测步=h+1, 预测价格=float(prices[h, 0]),
            训练截止月份=cfg['training_cutoff_month'], 验证截止月份=cfg['split_end_months']['val'],
            最新观测月份=str(frame.index[-1])) for h, month in enumerate(dates)])
        table.to_csv(output/'future_predictions.csv', index=False, encoding='utf-8-sig')
        result = dict(available=False, reason='未来实际价格未知，不计算MAPE或精度得分')
        print(table.to_string(index=False), flush=True)
    else:
        start, end = cfg['bounds'][mode]
        if len(frame) < end or str(frame.index[end-1]) != cfg['split_end_months'][mode]:
            raise ValueError('数据未覆盖保存的评价区间')
        ys = cfg['y_scaler']
        labels = frame[TARGET].to_numpy(dtype=float)[:, None]
        y = ((labels-np.asarray(ys['mean']))/np.asarray(ys['scale'])).astype(np.float32)
        dataset = LNGDataset(x, y, frame.index, cfg['features'], start, end, seq, horizon)
        result, actual, pred = evaluate(model, dataset, ys, device, model_name, batch_size)
        save_evaluation(output, model_name, cfg, dataset, result, actual, pred, prefix=mode)
        print(pd.DataFrame([metric_row(model_name, result)]).to_string(index=False), flush=True)
    save_json(output/'metadata.json', dict(model=model_name, region=cfg['region'], profile=cfg['feature_mode'],
        mode=mode, data_source=source_info(source), checkpoint_file=Path(checkpoint_path).name,
        model_version=saved.get('model_version'), model_kwargs=cfg['model_kwargs'],
        features=cfg['features'], training_cutoff_month=cfg['training_cutoff_month'],
        selection_end_month=cfg['split_end_months']['val'], latest_observed_month=str(frame.index[-1]),
        unit='元/立方米', preprocessing_fitted_until=cfg['training_cutoff_month']))
    return result


def predict_main(model_name, argv=None):
    _, global_cfg, runtime, run, _ = load_settings(model_name, argv, prediction=True)
    device = device_for(runtime['device'])
    if 'checkpoint' in run:
        tasks = [run['checkpoint']]
    else:
        tasks = [run['checkpoint_dir']/r/(p+'.pt') for r in run['regions'] for p in run['profiles']]
    if any(not path.is_file() for path in tasks):
        raise FileNotFoundError('配套权重不存在，请核对checkpoint_dir或checkpoint')
    directory = run_directory(run['output_dir'], run['mode'])
    for checkpoint_path in tasks:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        cfg = saved['config']
        output = directory/cfg['region']/cfg['feature_mode']
        predict_one(checkpoint_path, global_cfg, model_name, device, output, run['mode'], run['batch_size'])
    print(f'PREDICTION_COMPLETE model={model_name} mode={run["mode"]} output={directory}', flush=True)
