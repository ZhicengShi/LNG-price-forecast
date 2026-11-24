"""重载年度预算版配套权重，核对预测、指标及文件完整性。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from annual_runtime import ROOT, VERSION, load_checkpoint, predict_checkpoint, read_frame, resolve_device


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=VERSION+'交付核验')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/交付核验.json')
    args = parser.parse_args()
    device = resolve_device(args.device)
    tolerance = 1e-4 if device == 'cpu' else 1e-8
    torch.set_num_threads(2)
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
    release = json.loads((ROOT/'release.json').read_text(encoding='utf-8'))
    assert release['version'] == VERSION
    for name, expected in release['data_sha256'].items():
        assert sha256(ROOT/'data'/name)==expected, f'基准数据已改变 {name}'
    manifest = ROOT/'manifest.json'
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text(encoding='utf-8'))['files'].items():
            assert sha256(ROOT/name)==expected, f'交付文件已改变 {name}'
    predictions = pd.read_csv(ROOT/'reference/年度预算逐年预测.csv')
    key = ['budget_year','model','profile','region']
    assert len(predictions)==210 and not predictions.duplicated(key).any()
    assert predictions.version.eq(VERSION).all()
    assert predictions.actual.isna().equals(predictions.budget_year.eq(2026))
    expected_rows = predictions.set_index(key)
    frames = {region:read_frame(ROOT/'data'/f'{name}_处理后数据表.xlsx') for region,name in
              [('shanghai','上海'),('zhejiang','浙江'),('jiangsu','江苏')]}
    for row in predictions[predictions.budget_year<=2025].itertuples():
        series = frames[row.region].loc[f'{row.budget_year}-01':f'{row.budget_year}-12', '终端非居民天然气价格（分省）']
        assert len(series)==12 and abs(float(series.mean())-row.actual)<1e-10
    ape = 100*(predictions.predicted-predictions.actual).abs()/predictions.actual
    assert np.allclose(ape, predictions.ape_percent, atol=1e-8, equal_nan=True)
    metrics = pd.read_csv(ROOT/'reference/2020-2025分地区指标.csv').set_index(['region','profile','model'])
    recomputed = predictions[predictions.budget_year<=2025].groupby(['region','profile','model']).ape_percent.mean()
    assert np.allclose(metrics.mape_percent, recomputed.reindex(metrics.index), atol=1e-8)
    assert np.allclose(metrics.accuracy_score_percent, 100-metrics.mape_percent, atol=1e-8)
    index = json.loads((ROOT/'weights/index.json').read_text(encoding='utf-8'))
    assert len(index)==78
    errors = []
    for entry in index:
        path = ROOT/entry['path']
        assert sha256(path)==entry['sha256'], f'权重校验值不同 {entry["path"]}'
        checkpoint = load_checkpoint(path,entry['budget_year'])
        budget_year = checkpoint['budget_year']
        last = (budget_year-1-2015)*12+7
        assert checkpoint['train_origins']==list(range(5,last-16+1))
        values = predict_checkpoint(checkpoint, ROOT/'data', device)
        spec = checkpoint['spec']
        for region,value in zip(spec['regions'],values):
            keyvalue = (budget_year,spec['model'],spec['profile'],region)
            expected = expected_rows.loc[keyvalue,'predicted']
            difference = abs(float(value)-float(expected))
            assert difference<tolerance, f'预测不一致 {keyvalue} {difference}'
            errors.append(difference)
    report = dict(version=VERSION, device=device, status='通过', checkpoints=len(index),
                  reloaded_predictions=len(errors), reference_predictions=len(predictions), metric_groups=len(metrics),
                  budget_years=[2024,2025,2026], max_price_difference=max(errors), tolerance=tolerance,
                  torch_version=str(torch.__version__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
