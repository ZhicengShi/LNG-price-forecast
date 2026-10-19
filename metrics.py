"""统一分月MAPE、分月精度得分及三步平均；百分数口径。"""
import numpy as np
import pandas as pd

COLUMNS = ['模型', '第1个月MAPE（%）', '第1个月精度得分（%）',
           '第2个月MAPE（%）', '第2个月精度得分（%）',
           '第3个月MAPE（%）', '第3个月精度得分（%）',
           '平均MAPE（%）', '平均精度得分（%）']


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('真实值和预测值须形状一致、非空且有限')
    if (actual <= 0).any():
        raise ValueError('MAPE要求实际价格为正')
    value = float(np.mean(abs((predicted-actual)/actual))*100)
    return dict(MAPE_percent=value, accuracy_score_percent=100-value)


def report(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 3 or actual.shape[1:] != (3, 1):
        raise ValueError('评价数组须为（预测起点数，3，1）')
    result = metrics(actual, predicted)
    result.update(windows=len(actual), forecast_values=int(actual.size),
                  by_horizon={str(h+1): metrics(actual[:, h], predicted[:, h]) for h in range(3)})
    return result


def metric_row(model_name, result):
    values = [model_name]
    for h in range(1, 4):
        part = result['by_horizon'][str(h)]
        values.extend([part['MAPE_percent'], part['accuracy_score_percent']])
    average = float(np.mean([result['by_horizon'][str(h)]['MAPE_percent'] for h in range(1, 4)]))
    values.extend([average, 100-average])
    return dict(zip(COLUMNS, values))


def save_table(path, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, encoding='utf-8-sig', float_format='%.6f')
