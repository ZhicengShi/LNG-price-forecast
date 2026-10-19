"""重载两份融合模型权重，核对360条固定测试预测。"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import metric_row, report
from joint_emaformer.data import REGIONS, windows
from joint_emaformer._training import predict
from joint_emaformer.analysis import as_arrays
from joint_emaformer._runtime import (HERE, MODEL_NAME, read_config, configure, load_checkpoint, frame_for,
    load, csv, save, run_directory)


def verify(device='auto'):
    import pandas as pd
    records, tables = [], {}
    for entry in load(HERE/'reference/weights_index.json'):
        profile = entry['profile']
        cfg, shared, _, _ = read_config(HERE/f'configs/{profile}.yaml')
        actual_device = configure(shared, device)
        path = HERE/entry['path']
        model, saved = load_checkpoint(path, actual_device)
        frame, layout, versions = frame_for(shared, profile, shared['period']['test_end'], saved['layout'])
        assert layout == entry['input_layout']
        assert saved['preprocessing'] == entry['preprocessing']
        data = windows(frame, layout, saved['preprocessing'], saved['fit_end'], len(frame), saved['parameters']['seq_len'])
        forecast = predict(model, data, saved['preprocessing'])
        source = pd.read_csv(HERE/f'reference/fixed_test/{profile}/seed_42/predictions.csv')
        y, expected = as_arrays(source)
        tolerance = 1e-4 if actual_device.type == 'cpu' else 1e-6
        np.testing.assert_allclose(data.actual, y, rtol=0, atol=1e-12)
        np.testing.assert_allclose(forecast, expected, rtol=0, atol=tolerance)
        assert len(data.starts) == 20
        for i, region in enumerate(REGIONS):
            tables[(region, profile)] = metric_row(MODEL_NAME, report(data.actual[:, :, i:i+1], forecast[:, :, i:i+1]))
        records.append(dict(profile=profile, values=int(y.size), max_absolute_price_difference=float(abs(forecast-expected).max()),
                            price_tolerance=tolerance, training_cutoff=saved['training_cutoff_month'],
                            latest_target_month=str(frame.index[-1]), data_sources=versions))
    return records, tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', type=Path, default=HERE/'outputs/verification')
    cli = parser.parse_args(argv)
    records, tables = verify(cli.device)
    output = run_directory(cli.output_dir, 'verification')
    for (region, profile), row in tables.items():
        csv(output/f'{region}_{profile}_metrics.csv', [row])
    save(output/'verification.json', dict(passed=True, checkpoints=2, values=360, records=records))
    print(f'JOINT_VERIFY_PASS weights=2 values=360 output={output}', flush=True)


if __name__ == '__main__':
    main()
