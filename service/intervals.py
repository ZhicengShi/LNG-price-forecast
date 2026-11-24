"""同地区同预测步的已成熟残差经验预测区间。"""
import numpy as np
import pandas as pd

from CoRegionLng.analysis import prediction_intervals


def future_intervals(predictions, table, origin, reference_matches, settings, version):
    confidence = settings['confidence']
    minimum, window = settings['min_residuals'], settings['residual_window']
    historical = None
    if reference_matches and table is not None:
        # t是预测首月全局行号；目标行号小于t的残差才已成熟。
        mature = table[table.origin_index+table.horizon <= origin].copy()
        if not mature.empty:
            historical = prediction_intervals(mature, minimum, window, confidence)
    items = []
    for row in predictions:
        prior = None
        if reference_matches and table is not None:
            prior = table[(table.region == row['region']) & (table.horizon == row['horizon'])
                          & (table.origin_index+table.horizon <= origin)].sort_values('origin_index').tail(window)
        count = len(prior) if prior is not None else 0
        reason = ('历史数据与校准参考不一致' if not reference_matches else
                  '当前模型未登记匹配的回测校准记录' if table is None else
                  f'已成熟残差不足{minimum}个' if count < minimum else None)
        radius = None
        if reason is None:
            errors = abs(prior.prediction-prior.actual).to_numpy()
            rank = min(int(np.ceil((count+1)*confidence)), count)
            radius = float(np.sort(errors)[rank-1])
        past = None
        if historical is not None:
            past = historical[(historical.region == row['region']) & (historical.horizon == row['horizon'])
                              & historical.lower.notna()]
        coverage_count = len(past) if past is not None else 0
        latest = str(pd.PeriodIndex(prior.target_month, freq='M').max()) if count else None
        items.append(dict(region=row['region'], horizon=row['horizon'], target_month=row['target_month'],
            available=reason is None, lower=row['prediction']-radius if radius is not None else None,
            upper=row['prediction']+radius if radius is not None else None,
            calibration_count=count, latest_calibration_target_month=latest, reason=reason,
            historical_coverage_percent=float(past.covered.astype(float).mean()*100) if coverage_count else None,
            historical_coverage_count=coverage_count))
    return dict(method='matured_absolute_residual_order_statistic', confidence=confidence,
        calibration_basis='same_parameters_features_rolling_refit_protocol', calibration_version=version,
        min_residuals=minimum, residual_window=window,
        interpretation='此区间描述价格波动。', items=items)
