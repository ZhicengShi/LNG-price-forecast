"""重载全部配套权重，复核数据、预处理、逐条预测及统一分月指标表。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import load_global, prepare_data
from metrics import metric_row, save_table, COLUMNS, report
from runtime import ROOT, MODEL_FOLDERS, device_for, load_checkpoint, evaluate, save_json
from CoRegionLng.verify import verify as verify_joint


def verify(device='auto', output=None):
    device = device_for(device)
    torch.set_num_threads(4)
    config = load_global(ROOT/'config.yaml')
    index = json.loads((ROOT/'reference/weight_index.json').read_text(encoding='utf-8'))
    reference = pd.read_csv(ROOT/'reference/test_predictions.csv')
    expected_single = len(MODEL_FOLDERS) * 3 * 2
    assert len(index) == expected_single and len(reference) == expected_single * 60
    assert len({(r["model"], r["region"], r["profile"]) for r in index}) == expected_single
    output = Path(output or ROOT/'outputs/verification')
    output.mkdir(parents=True, exist_ok=True)
    records, tables = [], {}
    for entry in index:
        name, region, profile = [entry[key] for key in ['model', 'region', 'profile']]
        path = ROOT/entry['path']
        model, saved = load_checkpoint(path, name, device)
        data, frame, metadata = prepare_data(config, region, profile)
        cfg = saved['config']
        assert cfg['features'] == list(data.feature_names) and cfg['bounds'] == metadata['bounds']
        assert metadata['medians'] == cfg['medians']
        for key in ['x_scaler', 'y_scaler']:
            for part in ['mean', 'scale']:
                np.testing.assert_array_equal(cfg[key][part], metadata[key][part])
        result, actual, prediction = evaluate(model, data.test, cfg['y_scaler'], device, name, 16)
        part = reference[(reference.model == name) & (reference.region == region) & (reference.profile == profile)]
        assert len(data.test) == 20 and len(part) == 60
        y = part['真实价格'].to_numpy().reshape(20, 3, 1)
        p = part['预测价格'].to_numpy().reshape(20, 3, 1)
        np.testing.assert_allclose(actual, y, rtol=0, atol=1e-12)
        # 参考预测使用CUDA。cuDNN TF32使6个月LSTM与CPU存在约1.01e-4元的差异；
        # CPU LSTM采用2e-4绝对容差，其他CPU模型1e-4，CUDA仍为1e-6。
        tolerance = (2e-4 if name == 'LSTM' else 1e-4) if device.type == 'cpu' else 1e-6
        np.testing.assert_allclose(prediction, p, rtol=0, atol=tolerance)
        row = metric_row(name, result)
        original = metric_row(name, report(y, p))
        for h in range(1, 4):
            assert row[f'第{h}个月精度得分（%）'] == 100-row[f'第{h}个月MAPE（%）']
        assert list(row) == COLUMNS
        key = (region, profile)
        tables.setdefault(key, []).append(row)
        records.append(dict(model=name, region=region, profile=profile, values=int(actual.size),
            max_absolute_price_difference=float(abs(prediction-p).max()), absolute_price_tolerance=tolerance,
            average_MAPE=result['MAPE_percent'], reference_average_MAPE=entry['expected_MAPE'],
            average_MAPE_difference=float(row['平均MAPE（%）']-original['平均MAPE（%）']),
            training_cutoff=cfg['training_cutoff_month'], selection_cutoff=cfg['split_end_months']['val']))
        del model
    joint_records, joint_tables = verify_joint(str(device))
    for key, row in joint_tables.items():
        tables[key].append(row)
    for (region, profile), rows in tables.items():
        save_table(output/f'{region}_{profile}_metrics.csv', rows)
    manifest_count = 0
    if (ROOT/'manifest.json').is_file():
        manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
        for key, record in manifest['files'].items():
            assert (ROOT/key).is_file(), key
            manifest_count += 1
    result = dict(all_checks_passed=True, device=str(device), checkpoints=len(index)+2, predicted_values=len(reference)+360,
        processed_data_regions=3, metric_columns=COLUMNS, metric_tables=6,
        max_absolute_price_difference=max(row['max_absolute_price_difference'] for row in records+joint_records),
        metrics=records, joint_metrics=joint_records, manifest_files=manifest_count)
    save_json(output/'verification.json', result)
    print(f'VERIFY_PASS device={device} weights={len(index)+2} values={len(reference)+360} metric_tables=6 manifest_files={manifest_count}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    verify(args.device, args.output_dir)
