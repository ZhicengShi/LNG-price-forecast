"""滚动回测的分月、分年指标、时间块置信区间及历史残差预测区间。"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_emaformer.data import REGIONS, NAMES
from joint_emaformer._runtime import HERE, load, save, csv, resolve_path, metrics_rows, run_directory


def block_indices(n, length, repeats, seed):
    if not 1 <= length < n:
        raise ValueError('块长度须小于预测起点数')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-length+1, size=(repeats, int(np.ceil(n/length))))
    return (starts[..., None]+np.arange(length)).reshape(repeats, -1)[:, :n]


def metrics(actual, forecast):
    error = forecast-actual
    return dict(MAPE=float(abs(error/actual).mean()*100), score=float(100-abs(error/actual).mean()*100),
                MAE=float(abs(error).mean()), RMSE=float(np.sqrt((error**2).mean())))


def as_arrays(table):
    if table.empty or not np.isfinite(table[['actual', 'prediction']].to_numpy()).all() or (table.actual <= 0).any():
        raise ValueError('分析结果须非空、数值有限且实际价格为正')
    origins = sorted(table.origin_index.unique())
    if len(origins) > 1 and not np.all(np.diff(origins) == 1):
        raise ValueError('时间块分析要求连续预测起点')
    expected = pd.MultiIndex.from_product([sorted(table.origin_index.unique()), [1, 2, 3], REGIONS],
                                         names=['origin_index', 'horizon', 'region'])
    actual_index = pd.MultiIndex.from_frame(table[['origin_index', 'horizon', 'region']])
    if len(table) != len(expected) or actual_index.duplicated().any() or set(actual_index) != set(expected):
        raise ValueError('预测起点、地区或步长有缺失/重复')
    ordered = table.set_index(['origin_index', 'horizon', 'region']).loc[expected]
    return (ordered.actual.to_numpy().reshape(-1, 3, 3),
            ordered.prediction.to_numpy().reshape(-1, 3, 3))


def confidence_rows(actual, forecast, indices, profile, scope, length, confidence=.95):
    rows = []
    q = [(1-confidence)/2, (1+confidence)/2]
    for i, region in enumerate(REGIONS):
        for h in ['all', 1, 2, 3]:
            sel = slice(None) if h == 'all' else slice(h-1, h)
            y = actual[:, sel, i]
            error = forecast[:, sel, i]-y
            bootstrap = dict(MAPE=abs(error[indices]/y[indices]).mean(axis=(1, 2))*100,
                             MAE=abs(error[indices]).mean(axis=(1, 2)),
                             RMSE=np.sqrt((error[indices]**2).mean(axis=(1, 2))))
            bootstrap['score'] = 100-bootstrap['MAPE']
            point = metrics(y, forecast[:, sel, i])
            for key, values in bootstrap.items():
                lo, hi = np.quantile(values, q)
                rows.append(dict(scope=scope, profile=profile, region=region, horizon=str(h),
                                 metric=key, estimate=point[key], lower=float(lo), upper=float(hi),
                                 origins=len(actual), block_months=length, repeats=len(indices), confidence=confidence))
    return rows


def prediction_intervals(table, minimum=24, window=60, confidence=.95):
    """同地区、同预测步，仅用实际目标月已成熟的过去绝对残差。"""
    records = []
    for (region, horizon), group in table.groupby(['region', 'horizon'], sort=False):
        group = group.sort_values('origin_index')
        for row in group.itertuples():
            prior = group[group.origin_index+horizon <= row.origin_index].tail(window)
            count = len(prior)
            if count >= minimum:
                residuals = abs(prior.prediction-prior.actual).to_numpy()
                order = min(int(np.ceil((count+1)*confidence)), count)
                radius = float(np.sort(residuals)[order-1])
                lower, upper = row.prediction-radius, row.prediction+radius
                covered = lower <= row.actual <= upper
            else:
                lower = upper = None
                covered = None
            latest = int((prior.origin_index+prior.horizon-1).max()) if count else None
            if latest is not None and latest >= row.origin_index:
                raise AssertionError('使用未成熟残差')
            records.append(dict(profile=row.profile, region=region, horizon=int(horizon),
                                origin_index=int(row.origin_index), target_month=row.target_month,
                                actual=row.actual, prediction=row.prediction, lower=lower, upper=upper,
                                covered=covered, calibration_count=count, latest_calibration_target_index=latest))
    return pd.DataFrame(records)


def analyze(folder, output, settings, profiles, seed):
    protocol = load(Path(folder)/'protocol.json')
    selection_end = pd.Period(protocol['selection_cutoff_month'], freq='M')
    rows, annual, intervals, coverages, comparisons, unavailable = [], [], [], [], [], []
    tables, worst = {}, {}
    lengths = list(dict.fromkeys(settings['sensitivity_blocks']+[settings['block_months']]))
    for profile in profiles:
        table = pd.read_csv(Path(folder)/profile/f'seed_{seed}/predictions.csv')
        if set(table.profile) != {profile}:
            raise ValueError('预测文件中的变量方案与目录不一致')
        as_arrays(table)
        tables[profile] = table
        scopes = [('backtest', table), ('after_selection', table[table.history_cutoff >= str(selection_end)])]
        for scope, part in scopes:
            if part.empty:
                unavailable.append(dict(profile=profile, scope=scope, reason='无可用预测起点'))
                continue
            y, p = as_arrays(part)
            rows.extend(dict(范围=scope, **row) for row in metrics_rows(part))
            for length in lengths if scope == 'backtest' else [settings['block_months']]:
                if not 1 <= length < len(y):
                    unavailable.append(dict(profile=profile, scope=scope, block_months=length,
                                             reason='时间块长度须小于预测起点数'))
                    continue
                indices = block_indices(len(y), length, settings['bootstrap_repeats'], settings['bootstrap_seed'])
                intervals.extend(confidence_rows(y, p, indices, profile, scope, length, settings['confidence']))
        for year, group in table.groupby(table.first_forecast_month.str[:4]):
            annual.extend(dict(年份=int(year), **row) for row in metrics_rows(group))
        pi = prediction_intervals(table, settings['interval_min_residuals'], settings['interval_residual_window'], settings['confidence'])
        csv(Path(output)/f'{profile}_prediction_intervals.csv', pi)
        for region in REGIONS:
            for h in ['all', 1, 2, 3]:
                part = pi[(pi.region == region) & pi.lower.notna()]
                if h != 'all':
                    part = part[part.horizon == h]
                coverages.append(dict(profile=profile, region=region, horizon=str(h), calibrated_values=len(part),
                    coverage_percent=float(part.covered.astype(float).mean()*100) if len(part) else None,
                    mean_width=float((part.upper-part.lower).mean()) if len(part) else None))
            part = table[table.region == region].copy()
            part['APE'] = abs((part.prediction-part.actual)/part.actual)*100
            grouped = part.groupby(['origin_index', 'history_cutoff', 'first_forecast_month']).APE.mean()
            worst[f'{profile}_{region}'] = [dict(origin_index=int(k[0]), history_cutoff=k[1], first_forecast_month=k[2],
                                                average_MAPE=float(v)) for k, v in grouped.nlargest(5).items()]
    if set(tables) == {'all', 'selected'}:
        y, full = as_arrays(tables['all'])
        y2, selected = as_arrays(tables['selected'])
        if sorted(tables['all'].origin_index.unique()) != sorted(tables['selected'].origin_index.unique()):
            raise ValueError('两方案起点不同，无法成对比较')
        np.testing.assert_array_equal(y, y2)
        length = settings['block_months']
        if 1 <= length < len(y):
            indices = block_indices(len(y), length, settings['bootstrap_repeats'], settings['bootstrap_seed'])
            for i, region in enumerate(REGIONS):
                for h in ['all', 1, 2, 3]:
                    sel = slice(None) if h == 'all' else slice(h-1, h)
                    delta = (abs((full[:, sel, i]-y[:, sel, i])/y[:, sel, i])-
                             abs((selected[:, sel, i]-y[:, sel, i])/y[:, sel, i]))*100
                    q = [(1-settings['confidence'])/2, (1+settings['confidence'])/2]
                    lower, upper = np.quantile(delta[indices].mean(axis=(1, 2)), q)
                    comparisons.append(dict(region=region, horizon=str(h), full_minus_selected_MAPE=float(delta.mean()),
                        lower=float(lower), upper=float(upper), block_months=length,
                        interpretation='positive means selected has lower MAPE'))
    csv(Path(output)/'metrics.csv', rows)
    csv(Path(output)/'annual_metrics.csv', annual)
    if intervals:
        csv(Path(output)/'confidence_intervals.csv', intervals)
    if comparisons:
        csv(Path(output)/'paired_profile_differences.csv', comparisons)
    csv(Path(output)/'prediction_interval_coverage.csv', coverages)
    save(Path(output)/'worst_origins.json', worst)
    summary = dict(model='三地区融合EMAformer', metrics=rows, annual_metrics=annual,
                   confidence_intervals=intervals, paired_profile_differences=comparisons,
                   prediction_interval_coverage=coverages, unavailable=unavailable, settings=settings,
                   primary_seed=seed, input_protocol=protocol,
                   year_definition='year of first forecast month',
                   confidence_interpretation='pointwise moving-block-bootstrap intervals conditional on fixed features and parameters; no retraining within resamples',
                   prediction_interval_interpretation='empirical bands from past matured residuals, distinct from metric confidence intervals',
                   retrospective_earlier_origins=True)
    save(Path(output)/'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'configs/backtest.yaml')
    parser.add_argument('--run-dir', type=Path, help='回测根目录，须包含protocol.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--profile', nargs='+', choices=['all', 'selected'])
    parser.add_argument('--seed', type=int)
    cli = parser.parse_args(argv)
    path = cli.config.resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    settings = cfg['analysis']
    if not 0 < settings['confidence'] < 1 or settings['bootstrap_repeats'] < 1:
        raise ValueError('confidence须在(0,1)内，bootstrap_repeats须为正整数')
    folder = cli.run_dir.resolve() if cli.run_dir else resolve_path(settings['input_dir'], path)
    base = cli.output_dir.resolve() if cli.output_dir else resolve_path(settings['output_dir'], path)
    output = run_directory(base, 'analysis')
    protocol = load(folder/'protocol.json')
    profiles = cli.profile or protocol['profiles']
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in ('all', 'selected') for p in profiles):
        raise ValueError('profiles须包含不重复的all或selected')
    seed = cfg['primary_seed'] if cli.seed is None else cli.seed
    analyze(folder, output, settings, profiles, seed)
    print(f'ANALYSIS_COMPLETE output={output}', flush=True)


if __name__ == '__main__':
    main()

