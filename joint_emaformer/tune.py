"""读取search.yaml，在开发期搜索参数并输出可直接训练的配置。"""
import copy
import gc
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_emaformer.data import PreparedData
from joint_emaformer._training import train
from joint_emaformer._search import candidates
from joint_emaformer._runtime import (HERE, common_arguments, load, save, digest, configure,
    frame_for, cutoff_index, run_directory, resolve_path, assert_versions, validate_parameters)
from datasets import load_global


def search_profile(folder, profile, settings, frame, layout, device):
    output = folder/profile
    output.mkdir(exist_ok=True)
    chosen_path = output/'selected_parameters.json'
    if chosen_path.exists():
        return load(chosen_path)
    save(output/'input_layout.json', layout)
    prepared = PreparedData(frame.iloc[:settings['development_end']].copy(), layout, device)
    plan = load(folder/'search_plan.json')
    trials_path = output/'screening_trials.json'
    trials = load(trials_path) if trials_path.exists() else []
    begin = time.monotonic()
    for number, p in enumerate(plan, 1):
        if any(row['trial'] == number for row in trials):
            continue
        try:
            result, _ = train(prepared, p, settings['train_end'], settings['primary_seed'], settings['development_end'])
            row = dict(trial=number, parameters=p, status='complete', **result)
        except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
            row = dict(trial=number, parameters=p, status='failed', error=str(exc))
            gc.collect()
            torch.cuda.empty_cache()
        trials.append(row)
        save(trials_path, trials)
        done = [r for r in trials if r['status'] == 'complete']
        print(f'SCREEN {profile} {number}/{len(plan)} score={row.get("score")} best={min(r["score"] for r in done):.6f} elapsed={time.monotonic()-begin:.0f}s', flush=True)
    ranked = sorted([r for r in trials if r['status'] == 'complete'], key=lambda r: (r['score'], max(r['MAPE']), r['trial']))
    if len(ranked) < settings['finalists']:
        raise RuntimeError('成功候选数量不足')
    folds = [(settings['development_end']-settings['fold_months']*(settings['validation_folds']-i),
              settings['development_end']-settings['fold_months']*(settings['validation_folds']-i-1))
             for i in range(settings['validation_folds'])]
    confirmed = []
    for item in ranked[:settings['finalists']]:
        path = output/f'confirm_trial_{item["trial"]}.json'
        records = load(path) if path.exists() else []
        for fold, (fit_end, end) in enumerate(folds, 1):
            for seed in settings['training_seeds']:
                if any(r['fold'] == fold and r['seed'] == seed for r in records):
                    continue
                result, _ = train(prepared, item['parameters'], fit_end, seed, end)
                records.append(dict(fold=fold, **result))
                save(path, records)
                print(f'CONFIRM {profile} trial={item["trial"]} fold={fold} seed={seed} MAPE={result["MAPE"]} epoch={result["best_epoch"]}', flush=True)
        errors = np.asarray([r['MAPE'] for r in records]).mean(axis=0)
        confirmed.append(dict(trial=item['trial'], parameters=item['parameters'], records=records,
                              mean_region_MAPE=errors.tolist(), score=float(errors.mean())))
        save(output/'confirmed_candidates.json', confirmed)
    chosen = min(confirmed, key=lambda row: (row['score'], max(row['mean_region_MAPE']), row['trial']))
    latest = sorted([r for r in chosen['records'] if r['fold'] == settings['validation_folds']],
                    key=lambda r: (r['best_epoch'], r['seed']))
    representative = latest[len(latest)//2]
    schedule = [r['lr'] for r in representative['history'][:representative['best_epoch']]]
    chosen.update(frozen_epochs=len(schedule), frozen_lr_schedule=schedule,
                  schedule_source=dict(fold=representative['fold'], seed=representative['seed'],
                                       rule='median of latest fold best epochs across all three seeds'),
                  selection_end_month=str(frame.index[settings['development_end']-1]))
    save(chosen_path, chosen)
    (output/'selected_parameters.yaml').write_text(yaml.safe_dump(chosen, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print('SELECTED', profile, chosen['trial'], chosen['parameters'], 'epochs', len(schedule), flush=True)
    return chosen


def main(argv=None):
    parser = common_arguments(__doc__)
    parser.add_argument('--run-dir', type=Path, help='指定或恢复同一次搜索目录')
    parser.add_argument('--trials', type=int, help='覆盖候选数量，用于流程检查')
    cli = parser.parse_args(argv)
    config_path = cli.config.resolve() if cli.config else HERE/'configs/search.yaml'
    settings = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    shared_path = cli.data_config.resolve() if cli.data_config else resolve_path(settings['data_config'], config_path)
    shared = load_global(shared_path)
    device = configure(shared, cli.device)
    settings = copy.deepcopy(settings)
    if cli.trials is not None:
        settings['search_trials'] = cli.trials
    if settings['search_trials'] < settings['finalists'] or settings['finalists'] < 1:
        raise ValueError('搜索数量须不少于finalists，finalists须为正整数')
    if settings['validation_folds'] < 1 or settings['fold_months'] < 3:
        raise ValueError('验证折数须为正数，每折至少3个月')
    profiles = cli.profile or settings['profiles']
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in ('all', 'selected') for p in profiles):
        raise ValueError('profiles须包含不重复的all或selected')
    if not settings['training_seeds'] or len(set(settings['training_seeds'])) != len(settings['training_seeds']):
        raise ValueError('复核种子须非空且不重复')
    plan = candidates(settings)
    for p in plan:
        validate_parameters(p)
    frames = {}
    for profile in profiles:
        frames[profile] = frame_for(shared, profile, shared['period']['validation_end'])
    frame = frames[profiles[0]][0]
    settings['train_end'] = cutoff_index(frame, shared['period']['train_end'])
    settings['development_end'] = len(frame)
    minimum = settings['development_end']-settings['fold_months']*settings['validation_folds']
    if min(minimum, settings['train_end']) < max(p['seq_len'] for p in plan)+3:
        raise ValueError('历史不足以构造参数搜索的完整训练窗口')
    identity = dict(settings=settings, profiles=profiles, data_sha256=frames[profiles[0]][2],
                    device=str(device), torch_version=str(torch.__version__),
                    code_sha256={name: digest(HERE/name) for name in
                        ('model.py', '_emaformer.py', '_emaformer_layers.py', 'data.py', '_training.py',
                         '_search.py', '_runtime.py', 'tune.py')})
    base = cli.output_dir.resolve() if cli.output_dir else resolve_path(settings['output_dir'], config_path)
    folder = cli.run_dir.resolve() if cli.run_dir else run_directory(base, 'search')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'protocol.json').exists():
        if load(folder/'protocol.json') != identity:
            raise ValueError('恢复搜索时配置、代码、设备或数据版本发生变化')
    else:
        save(folder/'protocol.json', identity)
        save(folder/'search_plan.json', plan)
    for profile in profiles:
        frame, layout, versions = frames[profile]
        chosen = search_profile(folder, profile, settings, frame, layout, device)
        # 导出可直接传给train.py的完整配置，不覆盖随包提供的最终配置。
        final = yaml.safe_load((HERE/f'configs/{profile}.yaml').read_text(encoding='utf-8'))
        try:
            relative = Path(os.path.relpath(shared_path, folder/profile)).as_posix()
        except ValueError:
            relative = str(shared_path)
        final.update(data_config=relative, parameters=chosen['parameters'])
        final['training'].update(cutoff=shared['period']['validation_end'], seed=settings['primary_seed'],
                                 epochs=chosen['frozen_epochs'], lr_schedule=chosen['frozen_lr_schedule'],
                                 output_dir='training_outputs')
        final['prediction'].update(checkpoint='training_outputs/请填写实际运行目录/best.pt', output_dir='prediction_outputs')
        (folder/profile/'final_config.yaml').write_text(yaml.safe_dump(final, allow_unicode=True, sort_keys=False), encoding='utf-8')
        assert_versions(shared, versions)
    print(f'SEARCH_COMPLETE output={folder}', flush=True)


if __name__ == '__main__':
    main()

