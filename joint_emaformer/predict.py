"""加载三地区融合权重，复现测试或预测未来3个月。"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_emaformer.data import windows, fill_history, REGIONS
from joint_emaformer._training import predict
from joint_emaformer._runtime import (HERE, common_arguments, config_paths, read_config, configure,
    frame_for, cutoff_index, load_checkpoint, prediction_rows, save_prediction_results, csv,
    save, run_directory, resolve_path, assert_versions, digest)


def forecast_future(model, frame, saved):
    p, prep = saved['parameters'], saved['preprocessing']
    if len(frame) < p['seq_len']:
        raise ValueError('未来预测的历史长度不足')
    known = fill_history(frame, prep['medians'], prep['method']).iloc[-p['seq_len']:]
    x = (known.to_numpy(dtype=float)-np.asarray(prep['x_mean']))/np.asarray(prep['x_scale'])
    if not np.isfinite(x).all():
        raise ValueError('未来预测输入仍有缺失')
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(x.astype(np.float32))[None].to(device),
                       torch.tensor([len(frame)], device=device)).cpu().numpy()
    prices = output*np.asarray(prep['y_scale'])+np.asarray(prep['y_mean'])
    if not np.isfinite(prices).all():
        raise FloatingPointError('未来价格预测包含非有限值，请检查输入数据和权重')
    return [dict(profile=saved['profile'], region=region, target_month=str(frame.index[-1]+h+1),
                 horizon=h+1, prediction=float(prices[0, h, i]),
                 model_training_cutoff=saved['training_cutoff_month'],
                 latest_observed_month=str(frame.index[-1]))
            for h in range(3) for i, region in enumerate(REGIONS)]


def main(argv=None):
    parser = common_arguments(__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--mode', choices=['test', 'future'])
    cli = parser.parse_args(argv)
    if cli.checkpoint and (cli.config or cli.profile):
        raise ValueError('指定checkpoint时由权重确定变量方案，无需另选config/profile')
    paths = config_paths(cli)
    if cli.checkpoint:
        metadata = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
        paths = [HERE/'configs'/f'{metadata["profile"]}.yaml']
    directories = {}
    for path in paths:
        cfg, shared, path, _ = read_config(path, cli.data_config)
        device = configure(shared, cli.device)
        checkpoint_path = cli.checkpoint.resolve() if cli.checkpoint else resolve_path(cfg['prediction']['checkpoint'], path)
        model, saved = load_checkpoint(checkpoint_path, device)
        if saved['profile'] != cfg['profile']:
            raise ValueError('预测配置的变量方案与权重不一致')
        mode = cli.mode or cfg['prediction']['mode']
        shared = deepcopy(shared)
        shared['period']['start'] = saved['layout']['first_month']
        frame, layout, versions = frame_for(shared, saved['profile'],
                shared['period']['test_end'] if mode == 'test' else None, saved['layout'])
        if mode == 'test' and versions != saved['data_sha256']:
            raise ValueError('历史测试要求数据SHA256与配套权重一致；新数据请使用future或重新训练')
        base = cli.output_dir.resolve() if cli.output_dir else resolve_path(cfg['prediction']['output_dir'], path)
        if (base, mode) not in directories:
            directories[(base, mode)] = run_directory(base, mode)
        output = directories[(base, mode)]/saved['profile']
        output.mkdir(exist_ok=False)
        if mode == 'future':
            if str(frame.index[-1]) < max(saved['training_cutoff_month'], saved['selection_cutoff_month']):
                raise ValueError('最新观测月份不能早于模型训练或参数选择截止月')
            csv(output/'future_predictions.csv', forecast_future(model, frame, saved))
        else:
            start = max(saved['fit_end'], cutoff_index(frame, saved['selection_cutoff_month']))
            data = windows(frame, layout, saved['preprocessing'], start, len(frame), saved['parameters']['seq_len'])
            forecast = predict(model, data, saved['preprocessing'])
            rows = prediction_rows(frame, data.starts, data.actual, forecast, saved['profile'], saved['training_cutoff_month'])
            save_prediction_results(output, rows)
        assert_versions(shared, versions)
        save(output/'metadata.json', dict(model='三地区融合EMAformer', profile=saved['profile'],
             model_training_cutoff=saved['training_cutoff_month'], latest_observed_month=str(frame.index[-1]),
             selection_cutoff=saved['selection_cutoff_month'], data_sha256=versions,
             checkpoint_sha256=digest(checkpoint_path), price_unit='元/立方米', mode=mode))
        print(f'PREDICT_COMPLETE profile={saved["profile"]} mode={mode} output={output}', flush=True)


if __name__ == '__main__':
    main()
