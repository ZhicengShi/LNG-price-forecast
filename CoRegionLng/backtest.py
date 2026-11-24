"""冻结配置的全历史逐月重训；支持按完成起点恢复运行。"""
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CoRegionLng.data import PreparedData, windows, REGIONS
from CoRegionLng._training import train, predict, checkpoint
from CoRegionLng._runtime import (HERE, common_arguments, load, save, csv, configure, frame_for,
    read_config, cutoff_index, package_checkpoint, save_checkpoint, prediction_rows, metrics_rows,
    run_directory, resolve_path)
from datasets import load_global


def completed_origins(table, audit, origins):
    if table.empty:
        return set()
    required = {(h, r) for h in (1, 2, 3) for r in REGIONS}
    audited = {int(row['origin_index']) for row in audit}
    complete = set()
    for origin, group in table.groupby('origin_index'):
        if (origin in origins and origin in audited and len(group) == 9
                and set(zip(group.horizon, group.region)) == required
                and np.isfinite(group[['actual', 'prediction']].to_numpy()).all()):
            complete.add(int(origin))
    return complete


def main(argv=None):
    parser = common_arguments(__doc__)
    parser.add_argument('--run-dir', type=Path, help='指定或恢复同一次回测目录')
    parser.add_argument('--start', help='预测首月YYYY-MM，或common_earliest')
    parser.add_argument('--end', help='最后一个预测起点的预测首月YYYY-MM')
    cli = parser.parse_args(argv)
    config_path = cli.config.resolve() if cli.config else HERE/'configs/backtest.yaml'
    settings = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    shared_path = cli.data_config.resolve() if cli.data_config else resolve_path(settings['data_config'], config_path)
    shared = load_global(shared_path)
    device = configure(shared, cli.device)
    profiles = cli.profile or settings['profiles']
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in ('all', 'selected') for p in profiles):
        raise ValueError('profiles须包含不重复的all或selected')
    seeds = settings['training_seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('training_seeds须为不重复的非负整数')
    configurations, inputs = {}, {}
    for profile in profiles:
        path = resolve_path(settings['model_configs'][profile], config_path)
        cfg, _, _, _ = read_config(path, shared_path)
        if cfg['profile'] != profile:
            raise ValueError('model_configs与变量方案不一致')
        configurations[profile] = cfg
        inputs[profile] = frame_for(shared, profile, shared['period']['test_end'])
    # 统一起点取两个方案中的最长窗口；单独运行一个方案也沿用同一比较口径。
    both = [read_config(resolve_path(settings['model_configs'][p], config_path), shared_path)[0] for p in ('all', 'selected')]
    initial = max(c['parameters']['seq_len'] for c in both)+3
    frame = inputs[profiles[0]][0]
    start_month = cli.start or settings['start']
    start = initial if start_month == 'common_earliest' else cutoff_index(frame, start_month)-1
    end_month = cli.end or settings.get('end')
    last = len(frame)-3 if end_month is None else cutoff_index(frame, end_month)-1
    if start < initial or last < start or last > len(frame)-3:
        raise ValueError('起点范围不足以构造训练或未来3个月标签')
    origins = list(range(start, last+1))
    identity = dict(profiles=profiles, training_seeds=seeds, primary_seed=settings['primary_seed'],
        parameters=configurations, data_sources=inputs[profiles[0]][2],
        first_month=str(frame.index[0]), initial_train_months=initial, origins=len(origins),
        origin_indices=origins, first_history_cutoff=str(frame.index[start-1]),
        last_history_cutoff=str(frame.index[last-1]), last_target_month=str(frame.index[last+2]),
        selection_cutoff_month=shared['period']['validation_end'], retrospective_earlier_origins=True,
        protocol_version=2, device=str(device), torch_version=str(torch.__version__))
    base = cli.output_dir.resolve() if cli.output_dir else resolve_path(settings['output_dir'], config_path)
    folder = cli.run_dir.resolve() if cli.run_dir else run_directory(base, 'backtest')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'protocol.json').exists():
        if load(folder/'protocol.json') != identity:
            raise ValueError('恢复回测时配置、协议、设备或源文件信息发生变化，请新建运行目录')
    else:
        save(folder/'protocol.json', identity)
    for profile in profiles:
        cfg = configurations[profile]
        frame, layout, versions = inputs[profile]
        p, schedule = cfg['parameters'], cfg['training']['lr_schedule']
        for seed in seeds:
            output = folder/profile/f'seed_{seed}'
            output.mkdir(parents=True, exist_ok=True)
            path, audit_path = output/'predictions.csv', output/'training_audit.json'
            table = pd.read_csv(path) if path.exists() else pd.DataFrame()
            audit = load(audit_path) if audit_path.exists() else []
            completed = completed_origins(table, audit, origins)
            if origins[-1] in completed and not (output/'last.pt').is_file():
                completed.remove(origins[-1])
            records = table[table.origin_index.isin(completed)].to_dict('records') if not table.empty else []
            audit = [row for row in audit if row['origin_index'] in completed]
            pending = [t for t in origins if t not in completed]
            with tqdm(pending, desc=f'{profile} seed={seed}', initial=len(completed), total=len(origins),
                      disable=not sys.stderr.isatty(), file=sys.stderr, ncols=100, ascii=True) as progress:
                for t in progress:
                    started_at, started = datetime.now(timezone.utc).isoformat(), perf_counter()
                    prepared = PreparedData(frame.iloc[:t].copy(), layout, device)
                    model, prep, history = train(prepared, p, t, seed, replay_lrs=schedule)
                    ds = windows(frame.iloc[:t+3], layout, prep, t, t+3, p['seq_len'])
                    forecast = predict(model, ds, prep)
                    records.extend(prediction_rows(frame, [t], ds.actual, forecast, profile, str(frame.index[t-1])))
                    audit.append(dict(origin_index=t, history_cutoff=str(frame.index[t-1]), fit_end=t,
                        preprocessing_fit_end=prep['fit_end'], epochs=len(history), seed=seed,
                        training_windows=t-p['seq_len']-2, last_training_label_index=t-1,
                        schedule=schedule, training_loss=[row['training_loss'] for row in history],
                        started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=perf_counter()-started))
                    if t == origins[-1]:
                        saved = package_checkpoint(checkpoint(model, layout, p, prep, seed, schedule, t),
                                    profile, frame, versions, shared['period']['validation_end'])
                        save_checkpoint(output/'last.pt', saved)
                    records.sort(key=lambda row: (row['origin_index'], row['horizon'], REGIONS.index(row['region'])))
                    csv(path, records)
                    save(audit_path, sorted(audit, key=lambda row: row['origin_index']))
                    if (t-start+1) % 10 == 0 or t == last:
                        tqdm.write(f'BACKTEST profile={profile} seed={seed} completed={t-start+1}/{len(origins)} '
                                   f'cutoff={frame.index[t-1]}', file=sys.stderr)
            assert len(records) == len(origins)*9
            csv(output/'metrics.csv', metrics_rows(records))
    print(f'BACKTEST_COMPLETE output={folder}', flush=True)


if __name__ == '__main__':
    main()
