"""单区域训练共用流程；三个模型文件夹分别提供可直接运行的入口。"""
import copy
from datetime import datetime, timezone
import inspect
import random
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from datasets import TARGET, prepare_data
from metrics import metric_row
from runtime import (load_settings, device_for, get_model_class, run_directory, digest,
                     forward, evaluate, save_evaluation, save_json)


def seed_all(seed):
    # 与原单区域实验一致：DataLoader共享全局torch随机流。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(model, dataset, validation, y_scaler, name, device, params, save_best):
    """Adam+MSE，验证集MAPE早停；标签只来自训练分区。"""
    xs, ys = zip(*(dataset[i] for i in range(len(dataset))))
    training = TensorDataset(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
                             torch.tensor(list(dataset.starts)))
    loader = DataLoader(training, batch_size=params['batch_size'], shuffle=True, generator=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    best, stale, history = float('inf'), 0, []
    # 仅终端显示一层进度条；日志与进度条使用同一输出流，避免换行错位。
    log_stream = sys.stderr
    with tqdm(total=params['epochs'], desc='训练轮次', file=log_stream,
              disable=not log_stream.isatty(), ascii=True,
              ncols=min(100, shutil.get_terminal_size(fallback=(100, 24)).columns)) as progress:
        for epoch in range(1, params['epochs']+1):
            model.train()
            total = 0.0
            # tqdm.auto会建立DataLoader迭代器；隐藏批次进度但保留包装及原随机流。
            batches = tqdm(loader, disable=True)
            for x, y, starts in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(forward(model, x.to(device), starts.to(device), name), y.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('训练损失为非有限值')
                loss.backward()
                optimizer.step()
                total += loss.item()*len(x)
            validation_result, _, _ = evaluate(model, validation, y_scaler, device, name, params['batch_size'])
            score = validation_result['MAPE_percent']
            row = dict(epoch=epoch, train_MSE=total/len(training), val_MAPE_percent=score,
                       val_accuracy_score_percent=100-score)
            history.append(row)
            if score < best:
                best, stale = score, 0
                save_best(epoch, score)
            else:
                stale += 1
            progress.update(1)
            tqdm.write(f'epoch={epoch} train_MSE={row["train_MSE"]:.6f} val_MAPE={score:.6f}% best={best:.6f}%',
                       file=log_stream)
            log_stream.flush()
            if stale >= params['patience']:
                break
    return history


def train_one(name, global_cfg, model_cfg, runtime, params, region, profile, output):
    output.mkdir(parents=True, exist_ok=False)
    seed_all(runtime['seed'])
    data, frame, base = prepare_data(global_cfg, region, profile)
    device = device_for(runtime['device'])
    cls = get_model_class(name)
    defaults = {k: p.default for k, p in inspect.signature(cls).parameters.items()
                if p.default is not inspect.Parameter.empty}
    kwargs = dict(defaults, **model_cfg['parameters'])
    if {'input_size', 'seq_len', 'pred_len', 'target_index', 'target'}.intersection(model_cfg['parameters']):
        raise ValueError('输入维度、窗口长度与目标位置由数据配置确定')
    kwargs.update(input_size=len(data.feature_names), seq_len=data.train.seq_len, pred_len=data.train.pred_len)
    if 'target_index' in kwargs:
        kwargs['target_index'] = list(data.feature_names).index(TARGET)
    # from_dataset与原实验保持同一初始化及参数顺序。
    model = cls.from_dataset(data.train, **model_cfg['parameters']).to(device)
    cfg = dict(base, model=name, model_kwargs=kwargs, training=copy.deepcopy(params),
               seed=runtime['seed'])
    cfg['training'].pop('output_dir', None)
    for key in ('regions', 'profiles'):
        cfg['training'].pop(key, None)
    save_json(output/'config.json', cfg)
    source = global_cfg['data'][region]
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    def save_best(epoch, score):
        saved = dict(format_version='single_region_v1', model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                     config=cfg, best_epoch=epoch, val_MAPE_percent=score, data_sha256=base['data_sha256'])
        torch.save(saved, output/'best.pt')
    history = fit(model, data.train, data.val, cfg['y_scaler'], name, device, params, save_best)
    if digest(source) != base['data_sha256']:
        raise ValueError('训练期间源数据发生变化，请使用稳定数据版本重新运行')
    audit = dict(started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                 elapsed_seconds=perf_counter()-started, data_sha256=base['data_sha256'],
                 training_cutoff=base['training_cutoff_month'], validation_cutoff=base['split_end_months']['val'],
                 seed=runtime['seed'], device=str(device), epochs_run=len(history),
                 input_features=list(data.feature_names), parameters=kwargs, training_parameters=cfg['training'])
    best = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    best['training_audit'] = audit
    torch.save(best, output/'best.pt')
    model.load_state_dict(best['model_state_dict'], strict=True)
    result, actual, prediction = evaluate(model, data.test, cfg['y_scaler'], device, name, params['batch_size'])
    save_evaluation(output, name, cfg, data.test, result, actual, prediction)
    pd.DataFrame(history).to_csv(output/'history.csv', index=False, encoding='utf-8-sig')
    frame.to_csv(output/'input_snapshot.csv', encoding='utf-8-sig', index_label='年月')
    save_json(output/'preprocessing.json', {key: cfg[key] for key in ['features', 'missing', 'medians', 'x_scaler', 'y_scaler', 'training_cutoff_month']})
    save_json(output/'training_audit.json', audit)
    print(pd.DataFrame([metric_row(name, result)]).to_string(index=False), flush=True)
    return result


def train_main(model_name, argv=None):
    model_cfg, global_cfg, runtime, run, _ = load_settings(model_name, argv, prediction=False)
    for name in ['epochs', 'batch_size', 'patience']:
        if not isinstance(run[name], int) or isinstance(run[name], bool) or run[name] < 1:
            raise ValueError(f'{name}须为正整数')
    if not np.isfinite(run['lr']) or run['lr'] <= 0:
        raise ValueError('lr须为正数')
    directory = run_directory(run['output_dir'], 'train')
    for region in run['regions']:
        for profile in run['profiles']:
            output = directory/region/profile
            print(f'TRAIN_START model={model_name} region={region} profile={profile}', flush=True)
            train_one(model_name, global_cfg, model_cfg, runtime, run, region, profile, output)
    print(f'TRAIN_COMPLETE model={model_name} output={directory}', flush=True)
