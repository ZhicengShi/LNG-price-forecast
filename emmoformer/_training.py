"""EMMoFormer 的最终配置训练流程，共用正式数据读取和预测输出。"""
import copy
from datetime import datetime, timezone
import inspect
import shutil
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from datasets import TARGET, prepare_data
from emmoformer.model import EMMoFormer
from metrics import metric_row
from runtime import (device_for, evaluate, forward, load_settings,
                     run_directory, save_evaluation, save_json)
from training import seed_all


DERIVED_PARAMETERS = frozenset(('input_size', 'target_index', 'seq_len',
                                'pred_len', 'first_month_phase'))


def case_settings(settings, region, profile, epochs=None):
    """每个地区、变量方案读取自己的最终参数；命令行仅覆盖最大轮数。"""
    try:
        case = settings['experiments'][region][profile]
        parameters = copy.deepcopy(case['parameters'])
        params = copy.deepcopy(case['training'])
    except KeyError as error:
        raise ValueError(f'缺少 {region}/{profile} 的最终实验配置') from error
    if not isinstance(parameters, dict) or not isinstance(params, dict):
        raise ValueError('parameters 和 training 须为配置字典')
    if DERIVED_PARAMETERS.intersection(parameters):
        raise ValueError('输入维度、目标位置、窗口和首月相位由公共数据配置自动确定')
    if epochs is not None:
        params['epochs'] = epochs
    for name in ('epochs', 'patience', 'batch_size'):
        value = params.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name}须为正整数')
    for name in ('lr', 'grad_clip'):
        value = params.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name}须为有限正数')
    value = params.get('weight_decay')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError('weight_decay须为有限非负数')
    if params.get('optimizer') not in ('Adam', 'AdamW'):
        raise ValueError('optimizer须为Adam或AdamW')
    if params.get('loss') not in ('mse', 'mae', 'mape', 'huber'):
        raise ValueError('loss须为mse、mae、mape或huber')
    if params.get('scheduler') not in ('none', 'cosine'):
        raise ValueError('scheduler须为none或cosine')
    return parameters, params


def derive_model_kwargs(data, frame, base, parameters):
    """从本次实际输入派生维度和日历位置，检查窗口还原使用的目标尺度。"""
    if DERIVED_PARAMETERS.intersection(parameters):
        raise ValueError('输入维度、目标位置、窗口和首月相位不能手工覆盖')
    signature = inspect.signature(EMMoFormer)
    unknown = set(parameters).difference(signature.parameters)
    if unknown:
        raise ValueError(f'未知模型参数：{sorted(unknown)}')
    kwargs = {name: value.default for name, value in signature.parameters.items()
              if value.default is not inspect.Parameter.empty}
    kwargs.update(parameters)
    target_index = list(data.feature_names).index(TARGET)
    for key in ('mean', 'scale'):
        if not np.isclose(base['x_scaler'][key][target_index], base['y_scaler'][key][0],
                          rtol=0, atol=1e-12):
            raise ValueError('历史目标价格与预测标签的标准化参数须相同')
    kwargs.update(input_size=len(data.feature_names), target_index=target_index,
                  seq_len=data.train.seq_len, pred_len=data.train.pred_len,
                  first_month_phase=frame.index[0].month-1)
    return kwargs


def loss_value(prediction, target, params, scaler):
    """MAPE 在实际价格尺度计算，其余损失沿用标签的标准化尺度。"""
    name = params['loss']
    if name == 'mse':
        return torch.nn.functional.mse_loss(prediction, target)
    if name == 'mae':
        return (prediction-target).abs().mean()
    if name == 'huber':
        return torch.nn.functional.smooth_l1_loss(prediction, target)
    if name == 'mape':
        actual = target*scaler['scale'][0]+scaler['mean'][0]
        if (actual <= 0).any():
            raise ValueError('MAPE损失要求实际价格为正')
        return ((prediction-target).abs()*scaler['scale'][0]/actual).mean()
    raise ValueError('未知训练损失')


def fit(model, data, scaler, device, params, save_best):
    """按最终优化器、损失和学习率计划训练，以验证集 MAPE 保存最佳轮次。"""
    xs, ys = zip(*(data.train[i] for i in range(len(data.train))))
    dataset = TensorDataset(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
                            torch.tensor(list(data.train.starts)))
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    optimizer_class = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[params['optimizer']]
    optimizer = optimizer_class(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
                 if params['scheduler'] == 'cosine' else None)
    best, stale, history = float('inf'), 0, []
    stream = sys.stderr
    interactive = stream.isatty()
    with tqdm(total=params['epochs'], desc='训练轮次', file=stream,
              disable=not interactive, ascii=True,
              ncols=min(100, shutil.get_terminal_size(fallback=(100, 24)).columns)) as progress:
        for epoch in range(1, params['epochs']+1):
            model.train()
            total = 0.0
            for x, y, starts in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(model, x.to(device), starts.to(device), 'EMMoFormer')
                loss = loss_value(prediction, y.to(device), params, scaler)
                if not torch.isfinite(loss):
                    raise FloatingPointError('训练损失为非有限值')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(x)
            validation, _, _ = evaluate(model, data.val, scaler, device, 'EMMoFormer', batch_size=16)
            score = validation['MAPE_percent']
            history.append(dict(epoch=epoch, loss=total/len(dataset), validation_MAPE_percent=score,
                                lr=optimizer.param_groups[0]['lr']))
            if score < best:
                best, stale = score, 0
                save_best(epoch, score)
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step()
            if interactive:
                progress.set_postfix(loss=f'{history[-1]["loss"]:.6f}',
                                     val_MAPE=f'{score:.6f}%', best=f'{best:.6f}%', refresh=False)
                progress.update(1)
            else:
                print(f'epoch={epoch} loss={history[-1]["loss"]:.6f} '
                      f'val_MAPE={score:.6f}% best={best:.6f}%', file=stream, flush=True)
            if stale >= params['patience']:
                break
    return history


def train_one(global_cfg, experiment, runtime, region, profile, output, epochs=None):
    """训练一份地区及变量方案权重，保存配套预处理和测试结果。"""
    parameters, params = case_settings(experiment, region, profile, epochs)
    data, frame, base = prepare_data(global_cfg, region, profile)
    kwargs = derive_model_kwargs(data, frame, base, parameters)
    device = device_for(runtime['device'])
    output.mkdir(parents=True, exist_ok=False)
    seed_all(runtime['seed'])
    model = EMMoFormer(**kwargs).to(device)
    cfg = dict(base, model='EMMoFormer', model_kwargs=kwargs,
               training=copy.deepcopy(params), seed=runtime['seed'])
    save_json(output/'config.json', cfg)
    model_version = 'emmoformer-'+region+'-'+profile+'-'+uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()

    def save_best(epoch, score):
        torch.save(dict(format_version='single_region_v1', config=copy.deepcopy(cfg),
                        model_state_dict={key: value.detach().cpu().clone()
                                          for key, value in model.state_dict().items()},
                        best_epoch=epoch, val_MAPE_percent=score,
                        data_source=base['data_source'], model_version=model_version), output/'best.pt')

    history = fit(model, data, cfg['y_scaler'], device, params, save_best)
    best = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    audit = dict(started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                 elapsed_seconds=perf_counter()-started, data_source=base['data_source'],
                 training_cutoff=base['training_cutoff_month'],
                 validation_cutoff=base['split_end_months']['val'], seed=runtime['seed'],
                 device=str(device), epochs_run=len(history), best_epoch=best['best_epoch'],
                 validation_MAPE_percent=best['val_MAPE_percent'],
                 input_features=list(data.feature_names), parameters=kwargs,
                 training_parameters=cfg['training'],
                 parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                 training_windows=len(data.train), validation_windows=len(data.val))
    best['training_audit'] = audit
    torch.save(best, output/'best.pt')
    model.load_state_dict(best['model_state_dict'], strict=True)
    result, actual, prediction = evaluate(model, data.test, cfg['y_scaler'], device,
                                          'EMMoFormer', batch_size=16)
    save_evaluation(output, 'EMMoFormer', cfg, data.test, result, actual, prediction)
    pd.DataFrame(history).to_csv(output/'history.csv', index=False, encoding='utf-8-sig')
    frame.to_csv(output/'input_snapshot.csv', encoding='utf-8-sig', index_label='年月')
    save_json(output/'preprocessing.json', {key: cfg[key] for key in
              ('features', 'missing', 'medians', 'x_scaler', 'y_scaler', 'training_cutoff_month')})
    save_json(output/'training_audit.json', audit)
    print(pd.DataFrame([metric_row('EMMoFormer', result)]).to_string(index=False), flush=True)
    return result


def train_main(argv=None):
    settings, global_cfg, runtime, run, _ = load_settings('EMMoFormer', argv, prediction=False)
    tasks = [(region, profile) for region in run['regions'] for profile in run['profiles']]
    for region, profile in tasks:
        case_settings(settings, region, profile, run.get('epochs'))
    directory = run_directory(run['output_dir'], 'train')
    for region, profile in tasks:
        print(f'TRAIN_START model=EMMoFormer region={region} profile={profile}', flush=True)
        train_one(global_cfg, settings, runtime, region, profile, directory/region/profile,
                  epochs=run.get('epochs'))
    print(f'TRAIN_COMPLETE model=EMMoFormer output={directory}', flush=True)
