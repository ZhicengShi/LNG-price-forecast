"""筛选的时间隔离、窗口边界及结果输出检查。"""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import openpyxl
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'xgboost'))
from datasets import TARGET, load_global, read_frame
from selection import analyze, export_results, load_settings, selection_input, windows


class XGBoostSelectionTests(unittest.TestCase):
    def test_xgboost_keeps_twelve_months_when_neural_models_use_six(self):
        _, shared, args, _, _ = load_settings(ROOT/'xgboost/config.yaml')
        self.assertEqual(shared['window']['seq_len'], 6)
        self.assertEqual(args.seq_len, 12)
        self.assertEqual(args.pred_len, 3)

    def test_restored_combinations_are_validation_winners_and_match_training_config(self):
        import json
        shared = load_global(ROOT/'config.yaml')
        results = json.loads((ROOT/'xgboost/reference/selection.json').read_text(encoding='utf-8'))
        counts = dict(shanghai=4, zhejiang=3, jiangsu=5)
        for result in results:
            region = result['region']
            features = shared['features']['selected'][region]
            self.assertEqual(features, result['training_features'])
            self.assertEqual(len(features), counts[region])
            self.assertEqual(features[0], TARGET)
            best = result['combinations'][0]
            self.assertEqual(set(features), set(best['features']))
            self.assertEqual(result['parameters'], best['parameters'])
            self.assertAlmostEqual(result['val_MAPE'], best['val_MAPE'])

    def test_test_period_cannot_change_selection_inputs(self):
        shared = load_global(ROOT/'config.yaml')
        original = selection_input(shared, 'shanghai', 'all')
        full = read_frame(shared['data']['shanghai'])
        full.loc[full.index > pd.Period(shared['period']['validation_end'], freq='M')] = np.nan
        with patch('selection.read_frame', return_value=full):
            changed = selection_input(shared, 'shanghai', 'all')
        pd.testing.assert_frame_equal(original[0], changed[0])
        pd.testing.assert_frame_equal(original[1], changed[1])
        self.assertEqual(original[2:], changed[2:])
        self.assertEqual(str(original[0].index[-1]), '2024-07')

    def test_windows_fit_medians_only_to_past_and_keep_labels_in_partition(self):
        frame = pd.DataFrame({'leading_missing': [np.nan, np.nan, 10, 20, 30, 40, 50, 60, 70, 80],
                              TARGET: np.arange(1, 11, dtype=float)})
        original = windows(frame, 6, 0, 6, seq_len=2, horizon=3)
        altered = frame.copy()
        altered.iloc[6:] = 10000
        changed = windows(altered, 6, 0, 6, seq_len=2, horizon=3)
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[1], changed[1])
        self.assertEqual(original[0][0, 0], 25)
        np.testing.assert_array_equal(original[2], [2, 3])
        validation = windows(frame, 6, 6, 10, seq_len=2, horizon=3)
        np.testing.assert_array_equal(validation[2], [6, 7])
        self.assertEqual(validation[1][-1, -1], 10)
        frame.loc[7, TARGET] = np.nan
        with self.assertRaises(ValueError):
            windows(frame, 6, 6, 10, seq_len=2, horizon=3)

    def test_small_search_exports_monthly_metrics_without_changing_training_config(self):
        settings, shared, args, _, _ = load_settings(ROOT/'xgboost/config.yaml')
        settings, shared, args = copy.deepcopy(settings), copy.deepcopy(shared), copy.deepcopy(args)
        settings['candidate_features'] = ['TTF期货价格', 'LNG进口价格']
        shared['period'].update(train_end='2019-12', validation_end='2020-06')
        args.folds, args.fold_months, args.permutation_repeats = 2, 12, 2
        args.greedy_pool, args.greedy_max, args.seeds = 2, 2, [42]
        args.parameter_sets = {'depth2': dict(n_estimators=3, max_depth=2, learning_rate=.03)}
        before = (ROOT/'config.yaml').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            result = analyze('shanghai', shared, settings, args)
        self.assertFalse(result['selection_uses_test'])
        self.assertIn(TARGET, result['features'])
        self.assertEqual(result['training_features'][0], TARGET)
        with tempfile.TemporaryDirectory(prefix='lng_xgb_test_') as temporary:
            output = Path(temporary)
            export_results([result], output)
            selected = yaml.safe_load((output/'selected_features.yaml').read_text(encoding='utf-8'))
            self.assertEqual(selected['features']['selected']['shanghai'], result['training_features'])
            book = openpyxl.load_workbook(output/'上海_XGBoost变量筛选.xlsx', read_only=True, data_only=True)
            try:
                self.assertEqual(book.sheetnames, ['变量重要性', '组合比较', '最终变量'])
                rows = list(book['组合比较'].values)
                self.assertFalse(any('种子' in str(c) or 'seed' in str(c) for c in rows[0]))
                for row in rows[1:]:
                    values = dict(zip(rows[0], row))
                    monthly = []
                    for h in range(1, 4):
                        value = values[f'第{h}个月MAPE（%）']
                        self.assertIsInstance(value, (float, int))
                        self.assertAlmostEqual(values[f'第{h}个月精度得分（%）'], 100-value)
                        monthly.append(value)
                    self.assertAlmostEqual(values['平均MAPE（%）'], float(np.mean(monthly)))
                    self.assertAlmostEqual(values['平均精度得分（%）'], 100-float(np.mean(monthly)))
            finally:
                book.close()
        self.assertEqual(before, (ROOT/'config.yaml').read_bytes())


if __name__ == '__main__':
    unittest.main()
