"""按配置训练三地区融合EMAformer；默认分别训练全变量和组合变量。"""
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_emaformer.data import PreparedData, windows
from joint_emaformer._training import train, checkpoint, predict
from joint_emaformer._runtime import (common_arguments, config_paths, read_config, configure, frame_for,
    cutoff_index, package_checkpoint, save_checkpoint, load_checkpoint, prediction_rows,
    save_prediction_results, save, csv, run_directory, resolve_path)


def main(argv=None):
    parser = common_arguments(__doc__)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    cli = parser.parse_args(argv)
    directories = {}
    for path in config_paths(cli):
        cfg, shared, path, shared_path = read_config(path, cli.data_config)
        if cli.epochs is not None:
            if cli.epochs < 1:
                raise ValueError('epochs须为正整数')
            schedule = cfg['training']['lr_schedule']
            cfg['training'].update(epochs=cli.epochs, lr_schedule=(schedule+[schedule[-1]]*cli.epochs)[:cli.epochs])
        seed = cfg['training']['seed'] if cli.seed is None else cli.seed
        if seed < 0:
            raise ValueError('seed须为非负整数')
        device = configure(shared, cli.device)
        base = cli.output_dir.resolve() if cli.output_dir else resolve_path(cfg['training']['output_dir'], path)
        if base not in directories:
            directories[base] = run_directory(base, 'train')
        output = directories[base]/cfg['profile']
        output.mkdir(exist_ok=False)
        frame, layout, versions = frame_for(shared, cfg['profile'],
            None if cfg['training']['cutoff'] == 'latest' else shared['period']['test_end'])
        fit_end = cutoff_index(frame, cfg['training']['cutoff'])
        if fit_end < cfg['parameters']['seq_len']+3:
            raise ValueError('训练历史不足以构造完整窗口')
        started_at, started = datetime.now(timezone.utc).isoformat(), perf_counter()
        print(f'TRAIN_START profile={cfg["profile"]} cutoff={frame.index[fit_end-1]} device={device}', flush=True)
        schedule = cfg['training']['lr_schedule']
        model, prep, history = train(PreparedData(frame, layout, device), cfg['parameters'], fit_end, seed,
                                      replay_lrs=schedule, verbose=True)
        audit = dict(started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=perf_counter()-started, seed=seed, device=str(device),
                     data_sources=versions, training_cutoff_month=str(frame.index[fit_end-1]),
                     selection_cutoff_month=shared['period']['validation_end'],
                     epochs=len(history), input_layout=layout, parameters=cfg['parameters'],
                     configuration=cfg)
        saved = package_checkpoint(checkpoint(model, layout, cfg['parameters'], prep, seed, schedule, fit_end),
                                   cfg['profile'], frame, versions, shared['period']['validation_end'], audit)
        save_checkpoint(output/'best.pt', saved)
        save(output/'config.json', cfg)
        save(output/'preprocessing.json', prep)
        save(output/'training_audit.json', audit)
        csv(output/'history.csv', history)
        model, saved = load_checkpoint(output/'best.pt', device)
        start = max(fit_end, cutoff_index(frame, shared['period']['validation_end']))
        end = min(len(frame), cutoff_index(frame, shared['period']['test_end']))
        if end-start >= 3:
            data = windows(frame, layout, prep, start, end, cfg['parameters']['seq_len'])
            forecast = predict(model, data, prep)
            rows = prediction_rows(frame, data.starts, data.actual, forecast, cfg['profile'], saved['training_cutoff_month'])
            save_prediction_results(output, rows)
        else:
            save(output/'evaluation.json', dict(available=False, reason='训练截止月之后没有完整3个月测试标签'))
        print(f'TRAIN_COMPLETE profile={cfg["profile"]} output={output}', flush=True)


if __name__ == '__main__':
    main()
