"""三地区XGBoost重要性排序及验证集组合筛选；仅依赖本复现目录和已安装的库。"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import openpyxl
from openpyxl.styles import Alignment
import pandas as pd
import scipy
import xgboost
from xgboost import XGBRegressor
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets import TARGET, EXCLUDED_FEATURES, REGIONS, load_global, read_frame, resolve_path, selected_names
from provenance import source_info

HERE = Path(__file__).resolve().parent


def mape(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('指标形状不一致或含非有限值')
    if (actual <= 0).any():
        raise ValueError('目标价格必须为正数')
    return float(np.abs((predicted-actual)/actual).mean()*100)


def folds(train_end, months=15, count=3, seq_len=12, horizon=3):
    result = [(train_end-months*(count-i), train_end-months*(count-i-1)) for i in range(count)]
    if result[0][0] < seq_len+horizon or months < horizon:
        raise ValueError('训练区间不足以构造滚动验证')
    return result


def windows(frame, fit_end, start, end, seq_len=12, horizon=3, missing="ffill"):
    """时间连续；标签不跨分区；填充仅向前，中位数只由对应训练区间拟合。"""
    starts = np.arange(max(start, seq_len), end-horizon+1)
    if not len(starts):
        raise ValueError('区间不足以构造窗口')
    medians = frame.iloc[:fit_end].median()
    if missing not in ('ffill', 'median', 'error'):
        raise ValueError('缺失处理须为ffill、median或error')
    if missing == 'error' and frame.iloc[:end].isna().any().any():
        raise ValueError('输入存在缺失，missing=error不允许填充')
    values = (frame.ffill() if missing == 'ffill' else frame).fillna(medians).to_numpy(dtype=np.float32)
    x = np.stack([values[t-seq_len:t].reshape(-1) for t in starts])
    prices = frame[TARGET].to_numpy(dtype=float)
    y = np.stack([prices[t:t+horizon] for t in starts])
    if np.isinf(x).any() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('输入无穷或目标缺失/非正数')
    # 若某个早期排序折的训练列全空，保留NaN交给XGBoost；禁止借用后续中位数。
    return x, y, starts


def fit_models(x, y, parameters, seeds, base_parameters):
    return [[XGBRegressor(**base_parameters, **parameters, random_state=seed).fit(x, y[:,h])
             for h in range(y.shape[1])] for seed in seeds]


def predictions(models, x):
    return np.stack([np.column_stack([model.predict(x) for model in seed_models])
                     for seed_models in models])


def grouped_gain(models, variables, seq_len):
    rows = []
    for seed_models in models:
        for model in seed_models:
            values = np.zeros(variables*seq_len)
            for column, gain in model.get_booster().get_score(importance_type='total_gain').items():
                values[int(column[1:])] = gain
            grouped = values.reshape(seq_len, variables).sum(axis=0)
            rows.append(grouped/grouped.sum()*100 if grouped.sum() else grouped)
    return np.mean(rows, axis=0)


def block_order(length, block_size, rng):
    chunks = [np.arange(i, min(i+block_size, length)) for i in range(0, length, block_size)]
    if len(chunks) < 2:
        raise ValueError('置换至少需要两个时间块')
    order = rng.permutation(len(chunks))
    if np.array_equal(order, np.arange(len(chunks))):
        order = np.roll(order, 1)
    return np.concatenate([chunks[i] for i in order])


def grouped_permutation(models, x, y, variables, repeats, seed, block_size=3):
    """同一变量全部滞后一起置换，验证起点按配置的连续时间块打乱。"""
    rng = np.random.default_rng(seed)
    orders = np.stack([block_order(len(x), block_size, rng) for _ in range(repeats)])
    base = mape(y, predictions(models, x).mean(axis=0))
    result = []
    for variable in range(variables):
        columns = np.arange(variable, x.shape[1], variables)
        changed = np.repeat(x[None], repeats, axis=0)
        changed[:,:,columns] = x[orders][:,:,columns]
        forecast = predictions(models, changed.reshape(-1, x.shape[1])).mean(axis=0).reshape(repeats, *y.shape)
        result.append(np.abs((forecast-y[None])/y[None]).mean(axis=(1,2))*100-base)
    return np.asarray(result)


def rank_variables(frame, train_end, args):
    fold_rows, gains, permutation_means, cv_metrics = [], [], [], []
    names = list(frame.columns)
    for fold, (fit_end, eval_end) in enumerate(folds(train_end, args.fold_months, args.folds,
                                                   args.seq_len, args.pred_len), 1):
        x_train, y_train, _ = windows(frame, fit_end, 0, fit_end, args.seq_len, args.pred_len, args.missing)
        x_eval, y_eval, starts = windows(frame, fit_end, fit_end, eval_end, args.seq_len, args.pred_len, args.missing)
        models = fit_models(x_train, y_train, args.parameter_sets[args.ranking_model], args.seeds, args.base_parameters)
        gain = grouped_gain(models, len(names), args.seq_len)
        importance = grouped_permutation(models, x_eval, y_eval, len(names),
                                          args.permutation_repeats, args.permutation_seed+fold, args.permutation_block_months).mean(axis=1)
        gains.append(gain)
        permutation_means.append(importance)
        score = mape(y_eval, predictions(models, x_eval).mean(axis=0))
        cv_metrics.append(dict(fold=fold, fit_end=fit_end, eval_end=eval_end, MAPE=score,
                               windows=len(starts)))
        fold_rows.extend(dict(折=fold, 变量=name, 置换MAPE增加百分点=float(importance[i]),
                               总增益占比百分数=float(gain[i]),
                               训练末月=str(frame.index[fit_end-1]),
                               验证开始=str(frame.index[fit_end]), 验证结束=str(frame.index[eval_end-1]))
                         for i, name in enumerate(names))
        print(f'  排序折{fold}: {frame.index[fit_end]}~{frame.index[eval_end-1]}, MAPE={score:.4f}%', flush=True)
    gains, changes = np.asarray(gains), np.asarray(permutation_means)
    ranking = pd.DataFrame({'变量': names, '置换MAPE增加百分点': changes.mean(axis=0),
                            '跨折标准差百分点': changes.std(axis=0, ddof=1),
                            '正重要性折数': (changes>0).sum(axis=0),
                            '总增益占比百分数': gains.mean(axis=0)})
    ranking = ranking.sort_values(['置换MAPE增加百分点', '总增益占比百分数', '变量'],
                                  ascending=[False,False,True]).reset_index(drop=True)
    ranking.insert(0, '排序', np.arange(1,len(ranking)+1))
    return ranking, fold_rows, cv_metrics


def lag_columns(names, selected, seq_len):
    indexes = [names.index(name) for name in selected]
    return (np.arange(seq_len)[:,None]*len(names)+np.asarray(indexes)[None,:]).reshape(-1)


def select_combinations(frame, ranking, train_end, val_end, args):
    names = list(frame.columns)
    x_train, y_train, _ = windows(frame, train_end, 0, train_end, args.seq_len, args.pred_len, args.missing)
    x_val, y_val, _ = windows(frame, train_end, train_end, val_end, args.seq_len, args.pred_len, args.missing)
    permutation_order = ranking.loc[ranking['变量']!=TARGET, '变量'].tolist()
    gain_order = ranking.loc[ranking['变量']!=TARGET].sort_values(
        ['总增益占比百分数','变量'], ascending=[False,True])['变量'].tolist()
    cache = {}

    def evaluate(selected, parameter_name, method):
        selected = tuple(name for name in names if name in set(selected)|{TARGET})
        key = (selected, parameter_name)
        if key in cache:
            cache[key]['methods'].add(method)
            return cache[key]
        columns = lag_columns(names, selected, args.seq_len)
        models = fit_models(x_train[:,columns], y_train, args.parameter_sets[parameter_name], args.seeds, args.base_parameters)
        forecast = predictions(models, x_val[:,columns])
        scores = [mape(y_val, values) for values in forecast]
        ensemble = forecast.mean(axis=0)
        monthly = [mape(y_val[:, h], ensemble[:, h]) for h in range(args.pred_len)]
        row = dict(features=selected, parameters=parameter_name, methods={method},
                   val_by_horizon=monthly,
                   val_MAPE=mape(y_val,forecast.mean(axis=0)), seed_MAPE=scores,
                   seed_std=float(np.std(scores,ddof=1)) if len(scores)>1 else 0.)
        cache[key] = row
        if len(cache)%10==0:
            print(f'  已评估{len(cache)}个组合/参数候选，当前最佳验证MAPE={min(r["val_MAPE"] for r in cache.values()):.4f}%',flush=True)
        return row

    for parameter in args.parameter_sets:
        for method, order in [('置换排序TopK',permutation_order), ('增益排序TopK',gain_order)]:
            for count in range(len(order)+1):
                evaluate(order[:count], parameter, f'{method}({count})')
        current = []
        previous = evaluate(current,parameter,'前向选择(0)')['val_MAPE']
        pool = permutation_order[:args.greedy_pool]
        for count in range(1,min(args.greedy_max,len(pool))+1):
            candidates = [evaluate(current+[name],parameter,f'前向选择候选({count})')
                          for name in pool if name not in current]
            best = min(candidates,key=lambda row:(row['val_MAPE'],len(row['features']),row['features']))
            if best['val_MAPE'] >= previous-1e-9:
                break
            current = [name for name in best['features'] if name!=TARGET]
            best['methods'].add(f'前向选择路径({count})')
            previous = best['val_MAPE']
    ordered = sorted(cache.values(), key=lambda row:(row['val_MAPE'],len(row['features']),row['parameters'],row['features']))
    return ordered, ordered[0]


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def load_settings(path, data_config=None):
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    shared_path = Path(data_config).resolve() if data_config else resolve_path(settings['data_config'], path)
    shared = load_global(shared_path)
    ranking, combination = settings['ranking'], settings['combination']
    args = SimpleNamespace(**ranking, **combination, ranking_model=ranking['model'],
                           seq_len=shared['window']['seq_len'], pred_len=shared['window']['pred_len'],
                           missing=shared['preprocessing']['missing'], seeds=settings['seeds'],
                           parameter_sets=settings['parameters']['variants'],
                           base_parameters=settings['parameters']['base'])
    for key in ('folds', 'fold_months', 'permutation_repeats', 'permutation_block_months',
                'greedy_pool', 'greedy_max', 'seq_len', 'pred_len'):
        value = getattr(args, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{key}须为正整数')
    if args.folds < 2 or args.pred_len != 3:
        raise ValueError('排序至少2折，当前分月指标要求预测3个月')
    if (not isinstance(args.seeds, list) or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in args.seeds)):
        raise ValueError('seeds须为非空、不重复的非负整数列表')
    if not isinstance(args.permutation_seed, int) or args.permutation_seed < 0:
        raise ValueError('permutation_seed须为非负整数')
    if not args.parameter_sets or args.ranking_model not in args.parameter_sets:
        raise ValueError('ranking.model须对应一个parameters.variants参数方案')
    for name, parameters in args.parameter_sets.items():
        if set(parameters) & set(args.base_parameters) or 'random_state' in parameters:
            raise ValueError(f'{name}与base包含重复参数，或重复指定random_state')
    if 'random_state' in args.base_parameters:
        raise ValueError('random_state由seeds指定')
    if args.missing not in ('ffill', 'median', 'error'):
        raise ValueError('缺失处理须为ffill、median或error')
    return settings, shared, args, path, shared_path


def selection_input(shared, region, candidates):
    """只截取训练和验证月份；测试数据不进入排序或组合评价。"""
    source = shared['data'][region]
    full = read_frame(source)
    period = shared['period']
    for key in ('start', 'train_end', 'validation_end'):
        if not re.fullmatch(r'\d{4}-\d{2}', str(period[key])):
            raise ValueError(f'period.{key}须使用YYYY-MM')
    first, train_last, val_last = [pd.Period(period[k], freq='M') for k in
                                  ('start', 'train_end', 'validation_end')]
    if not first <= train_last < val_last:
        raise ValueError('筛选训练和验证月份须按时间顺序排列')
    months = pd.period_range(first, val_last, freq='M')
    if not months.isin(full.index).all():
        raise ValueError('数据未覆盖筛选训练和验证月份')
    raw = full.loc[months].copy()
    train_end = raw.index.get_loc(train_last)+1
    target = raw[TARGET].to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError('筛选区间目标价格须为正且不可缺失，不允许填充标签')
    choice = candidates[region] if isinstance(candidates, dict) else candidates
    names, _ = selected_names(raw, {'features': {'all': choice}}, region, 'all', train_end)
    excluded = []
    for name in raw:
        if name in names:
            continue
        reason = ('地区停用字段' if name in EXCLUDED_FEATURES[REGIONS[region]] else
                  '训练期全空' if raw[name].iloc[:train_end].isna().all() else '未列入候选')
        excluded.append(dict(变量=name, 状态=reason))
    return raw, raw[names], train_end, excluded, source_info(source)


def analyze(region, shared, settings, args):
    started_at, started = datetime.now(timezone.utc).isoformat(), perf_counter()
    raw, frame, train_end, excluded, source = selection_input(shared, region, settings['candidate_features'])
    if train_end < args.seq_len+args.pred_len or len(frame)-train_end < args.pred_len:
        raise ValueError('训练或验证区间不足以构造窗口')
    print(f'{REGIONS[region]}: {len(frame.columns)}个候选变量（含历史目标），开始排序', flush=True)
    ranking, fold_rows, cv = rank_variables(frame, train_end, args)
    combinations, chosen = select_combinations(frame, ranking, train_end, len(frame), args)
    candidate_order = ranking.loc[ranking['变量'] != TARGET, '变量'].tolist()
    gain_order = ranking.sort_values(['总增益占比百分数', '变量'], ascending=[False, True])['变量'].tolist()
    ranking['候选变量排序'] = ranking['变量'].map({name: i+1 for i, name in enumerate(candidate_order)})
    ranking['增益排序'] = ranking['变量'].map({name: i+1 for i, name in enumerate(gain_order)})
    ranking['状态'] = '可用'
    complete = ranking.to_dict('records')+excluded
    for row in complete:
        name = row['变量']
        row['历史目标强制保留'] = '是' if name == TARGET else '否'
        row['推荐组合保留'] = '是' if name in chosen['features'] else '否'
        row['筛选区间缺失月份数'] = int(raw[name].isna().sum())
        row['训练区间缺失月份数'] = int(raw[name].iloc[:train_end].isna().sum())
    result = dict(region=region, label=REGIONS[region], features=list(chosen['features']),
                  # 训练配置沿用历史目标第一列，外生变量保持原始表顺序。
                  training_features=[TARGET, *[name for name in chosen['features'] if name != TARGET]],
                  parameters=chosen['parameters'], val_MAPE=chosen['val_MAPE'],
                  val_by_horizon=chosen['val_by_horizon'], candidates=len(combinations),
                  ranking=complete, ranking_folds=fold_rows, cv_metrics=cv, combinations=combinations,
                  source_file=shared['data'][region].name, data_source=source,
                  dates=dict(start=str(frame.index[0]), train_end=str(frame.index[train_end-1]),
                             validation_start=str(frame.index[train_end]), validation_end=str(frame.index[-1])),
                  train_months=train_end, validation_months=len(frame)-train_end,
                  validation_windows=len(frame)-train_end-args.pred_len+1, selection_uses_test=False,
                  settings=vars(args), versions=dict(python=sys.version.split()[0], xgboost=xgboost.__version__,
                                                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
                  started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=perf_counter()-started)
    print(f'{REGIONS[region]}: 完成{len(combinations)}组候选，验证MAPE={chosen["val_MAPE"]:.6f}%，'
          f'入选{len(chosen["features"])}个变量（含历史目标）', flush=True)
    return json_value(result)


def monthly_row(monthly):
    row = {}
    for h, value in enumerate(monthly, 1):
        row[f'第{h}个月MAPE（%）'] = value
        row[f'第{h}个月精度得分（%）'] = 100-value
    average = float(np.mean(monthly))
    row.update({'平均MAPE（%）': average, '平均精度得分（%）': 100-average})
    return row


def matrix(rows, columns=None):
    frame = pd.DataFrame(rows)
    if columns is not None:
        frame = frame.reindex(columns=columns)
    return json_value([list(frame.columns), *frame.astype(object).where(frame.notna(), None).values.tolist()])


def workbook_tables(result):
    """三个普通数据工作表；数值保持数值类型，种子仅在运行元数据中保存。"""
    rank_columns = ['排序', '候选变量排序', '变量', '状态', '历史目标强制保留', '推荐组合保留',
                    '置换MAPE增加百分点', '跨折标准差百分点', '正重要性折数', '总增益占比百分数',
                    '增益排序', '训练区间缺失月份数', '筛选区间缺失月份数']
    comparison = []
    for i, row in enumerate(result['combinations'], 1):
        comparison.append(dict(组合排名=i, 推荐='是' if i == 1 else '否',
                               变量数含历史价格=len(row['features']), 参数方案=row['parameters'],
                               **monthly_row(row['val_by_horizon']),
                               组合方法='；'.join(row['methods']), 变量组合='；'.join(row['features'])))
    final = [['序号', '入选变量', '历史目标强制保留'],
             *[[i, name, '是' if name == TARGET else '否'] for i, name in enumerate(result['training_features'], 1)],
             [None, None, None], ['项目', '数值或说明', None]]
    settings = result['settings']
    explanations = [
        ['地区', result['label']], ['数据文件', result['source_file']],
        ['训练区间', result['dates']['start']+'至'+result['dates']['train_end']],
        ['组合选择验证区间', result['dates']['validation_start']+'至'+result['dates']['validation_end']],
        ['验证预测起点数', result['validation_windows']], ['输入历史月数', settings['seq_len']],
        ['预测月数', settings['pred_len']], ['入选参数方案', result['parameters']],
        ['最佳组合验证MAPE（%）', result['val_MAPE']],
        ['评估组合与参数数量', result['candidates']],
        ['排序方式', f'训练期内{settings["folds"]}折扩展窗口，每折验证{settings["fold_months"]}个月'],
        ['置换方式', f'同一变量全部滞后成组；{settings["permutation_block_months"]}个月时间块；重复{settings["permutation_repeats"]}次'],
        ['选择规则', '按多种子平均预测的验证MAPE选择；仅在本次候选范围内最优'],
        ['计分方式', '各起点、三个预测步等权；各月得分=100-MAPE；平均值为三步等权平均'],
        ['测试数据是否参与选择', '否'], ['历史目标价格', '始终保留；未来标签不填充'],
        ['缺失处理', settings['missing']+'；中位数仅拟合对应训练历史，早期训练全空列保留NaN'],
        ['标准化', 'XGBoost使用原始数值，不拟合标准化参数'],
        ['指标范围', '组合比较表均为XGBoost验证指标'],
        ['重要性含义', '置换MAPE增加越大，当前模型越依赖该变量；负值保留'],
        ['跨折标准差', '描述历史折之间的变化，非置信区间'],
        ['解释范围', '当前XGBoost的预测贡献，不是因果效应或神经网络的重要性'],
        ['XGBoost版本', result['versions']['xgboost']],
    ]
    final.extend([[key, value, None] for key, value in explanations])
    return [dict(name='变量重要性', rows=matrix(result['ranking'], rank_columns)),
            dict(name='组合比较', rows=matrix(comparison)), dict(name='最终变量', rows=final)]


def export_results(results, output):
    """独立运行所需的XLSX导出，仅使用requirements中的Python库。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for result in results:
        book = openpyxl.Workbook()
        book.remove(book.active)
        for table in workbook_tables(result):
            sheet = book.create_sheet(table['name'])
            for row in table['rows']:
                sheet.append(row)
            # 无底色、装饰边框或字体设置；仅调整列宽及长文本换行。
            for column in sheet.columns:
                letter = column[0].column_letter
                width = max(sum(2 if ord(c) > 127 else 1 for c in str(cell.value or '')) for cell in column[:9])+2
                sheet.column_dimensions[letter].width = min(70, max(12, width))
                for cell in column:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
                    if isinstance(cell.value, float):
                        cell.number_format = '0.000000'
            for row in sheet:
                lines = 1
                for cell in row:
                    width = max(1, sheet.column_dimensions[cell.column_letter].width-2)
                    text = str(cell.value if cell.value is not None else '')
                    cell_lines = sum(max(1, math.ceil(sum(2 if ord(c) > 127 else 1 for c in line)/width))
                                     for line in text.split('\n'))
                    lines = max(lines, cell_lines)
                sheet.row_dimensions[row[0].row].height = max(32 if row[0].row == 1 else 18, lines*15+3)
        book.save(output/f'{result["label"]}_XGBoost变量筛选.xlsx')
        book.close()
    selected = {'features': {'selected': {r['region']: r['training_features'] for r in results}}}
    (output/'selected_features.yaml').write_text(yaml.safe_dump(selected, allow_unicode=True, sort_keys=False), encoding='utf-8')
    save_json(output/'selection.json', results)


def check_reference(result, expected):
    """核对原实验的排序和全部组合，分月指标由本次重算补充。"""
    if set(result['features']) != set(expected['features']) or result['parameters'] != expected['parameters']:
        raise AssertionError(f'{result["region"]}最终变量或参数与参考实验不一致')
    if result['candidates'] != expected['candidates']:
        raise AssertionError('候选组合数量不同')
    np.testing.assert_allclose(result['val_MAPE'], expected['val_MAPE'], rtol=0, atol=1e-9)
    ranked = [row for row in result['ranking'] if row.get('状态') == '可用']
    if [row['变量'] for row in ranked] != [row['变量'] for row in expected['ranking']]:
        raise AssertionError('变量重要性顺序不同')
    for key in ('置换MAPE增加百分点', '跨折标准差百分点', '总增益占比百分数'):
        np.testing.assert_allclose([row[key] for row in ranked], [row[key] for row in expected['ranking']], rtol=0, atol=1e-9)
    for actual, original in zip(result['combinations'], expected['combinations']):
        if actual['features'] != original['features'] or actual['parameters'] != original['parameters']:
            raise AssertionError('候选组合顺序或内容不同')
        np.testing.assert_allclose(actual['val_MAPE'], original['val_MAPE'], rtol=0, atol=1e-9)
    return dict(region=result['region'], passed=True, candidates=result['candidates'],
                features_match=True, ranking_match=True, validation_metrics_match=True,
                source_file=result['source_file'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.yaml')
    parser.add_argument('--data-config', type=Path)
    parser.add_argument('--region', nargs='+', choices=list(REGIONS))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-reference', action='store_true', help='核对原数据版本的三地区参考筛选结果')
    cli = parser.parse_args(argv)
    settings, shared, args, config_path, shared_path = load_settings(cli.config, cli.data_config)
    regions = cli.region if cli.region is not None else settings['regions']
    if not regions or len(set(regions)) != len(regions) or any(r not in REGIONS for r in regions):
        raise ValueError('regions须包含不重复的有效地区')
    output_base = cli.output_dir.resolve() if cli.output_dir else resolve_path(settings['output_dir'], config_path)
    output_base.mkdir(parents=True, exist_ok=True)
    stem, number = 'selection_'+datetime.now().strftime('%Y%m%d_%H%M'), 1
    while True:
        output = output_base/(stem if number == 1 else f'{stem}_{number}')
        try:
            output.mkdir()
            break
        except FileExistsError:
            number += 1
    expected = {}
    if cli.check_reference:
        expected = {r['region']: r for r in json.loads((HERE/'reference/original_selection.json').read_text(encoding='utf-8'))}
    results, checks = [], []
    for region in regions:
        result = analyze(region, shared, settings, args)
        if cli.check_reference:
            checks.append(check_reference(result, expected[region]))
        results.append(result)
        # 每地区完成即保存；长任务中断后已完成地区的结果仍可核对。
        export_results(results, output)
    save_json(output/'run.json', dict(config=settings, data_config=shared,
              config_file=config_path.name, data_config_file=shared_path.name,
              reference_checks=checks, selected_regions=regions, selection_uses_test=False))
    print(f'SELECTION_COMPLETE output={output}', flush=True)


if __name__ == '__main__':
    main()


