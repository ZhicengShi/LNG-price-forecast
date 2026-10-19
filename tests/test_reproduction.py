"""时间隔离、周期位置和分月精度得分的必要边界检查。"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets import load_global, prepare_data, read_frame, TARGET
from metrics import report, metric_row, COLUMNS
from runtime import forward, load_inputs


class ReproductionTests(unittest.TestCase):
    def test_future_data_does_not_change_training_windows_or_preprocessing(self):
        cfg = load_global(ROOT/'config.yaml')
        original, _, base = prepare_data(cfg, 'shanghai', 'selected')
        frame = read_frame(cfg['data']['shanghai'])
        changed = frame.copy()
        changed.iloc[95:] = changed.iloc[95:]*4+1
        with patch('datasets.read_frame', return_value=changed):
            altered, _, altered_meta = prepare_data(cfg, 'shanghai', 'selected')
        self.assertEqual(base['medians'], altered_meta['medians'])
        self.assertEqual(base['x_scaler'], altered_meta['x_scaler'])
        self.assertEqual(base['y_scaler'], altered_meta['y_scaler'])
        for i in [0, len(original.train)-1]:
            np.testing.assert_array_equal(original.train[i][0], altered.train[i][0])
            np.testing.assert_array_equal(original.train[i][1], altered.train[i][1])

    def test_phase_is_forecast_first_month_for_ema_and_last_history_for_mofo(self):
        class Phase:
            cycle_len = 24
            periodic = 12
            def __call__(self, x, value):
                return value
        starts = torch.tensor([12, 23, 24, 137])
        model = Phase()
        np.testing.assert_array_equal(forward(model, None, starts, 'EMAformer'), [12, 23, 0, 17])
        np.testing.assert_array_equal(forward(model, None, starts, 'MoFo'), [11, 10, 11, 4])

    def test_each_month_has_its_own_score_and_average_is_consistent(self):
        actual = np.ones((2, 3, 1))
        predicted = actual+np.array([.01, .02, .03])[None, :, None]
        row = metric_row('LSTM', report(actual, predicted))
        self.assertEqual(list(row), COLUMNS)
        for h in [1, 2, 3]:
            self.assertAlmostEqual(row[f'第{h}个月MAPE（%）'], h)
            self.assertAlmostEqual(row[f'第{h}个月精度得分（%）'], 100-h)
        self.assertAlmostEqual(row['平均MAPE（%）'], 2)
        self.assertAlmostEqual(row['平均精度得分（%）'], 98)
        bad = metric_row('LSTM', report(actual, actual*3))
        self.assertEqual(bad['平均精度得分（%）'], -100)

    def test_target_missing_is_rejected_in_training_and_prediction(self):
        cfg = load_global(ROOT/'config.yaml')
        _, frame, metadata = prepare_data(cfg, 'shanghai', 'selected')
        frame.loc[frame.index[-1], TARGET] = np.nan
        with patch('datasets.read_frame', return_value=frame):
            with self.assertRaises(ValueError):
                prepare_data(cfg, 'shanghai', 'selected')
        with self.assertRaises(ValueError):
            load_inputs(frame, metadata)

    def test_training_iterator_preserves_archived_first_epoch(self):
        from training import train_one
        torch.set_num_threads(4)
        cfg = load_global(ROOT/'config.yaml')
        model_cfg = yaml.safe_load((ROOT/'lstm/config.yaml').read_text(encoding='utf-8'))
        params = dict(model_cfg['training'], epochs=1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        runtime = dict(cfg['runtime'], device=device)
        with tempfile.TemporaryDirectory(prefix='lng_reproduction_') as temporary:
            output = Path(temporary)/'training'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                train_one('LSTM', cfg, model_cfg, runtime, params, 'shanghai', 'selected', output)
            history = pd.read_csv(output/'history.csv')
            # 源实验第一轮；删除tqdm包装会改变DataLoader随机流，MSE约变为1.0630。
            if device == 'cuda':
                self.assertAlmostEqual(float(history.iloc[0].train_MSE), 1.0654219874629267, delta=1e-4)
            else:
                # CPU与CUDA的dropout随机实现不同；CPU只核验训练流程完整性。
                self.assertTrue(np.isfinite(float(history.iloc[0].train_MSE)))
            saved = torch.load(output/'best.pt', weights_only=True, map_location='cpu')
            self.assertEqual(saved['best_epoch'], 1)
            self.assertIn('training_audit', saved)
            self.assertEqual(saved['config']['training_cutoff_month'], '2022-11')
            self.assertEqual(list(pd.read_csv(output/'test_metrics.csv')), COLUMNS)


if __name__ == '__main__':
    unittest.main()
