"""EMMoFormer 的日历、配置派生、最终权重和训练预测入口检查。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets import TARGET, load_global, prepare_data
from emmoformer._training import (DERIVED_PARAMETERS, case_settings,
                                  derive_model_kwargs, loss_value, train_one)
from runtime import forward, load_checkpoint, predict_one


class EMMoFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.global_cfg = load_global(ROOT/'config.yaml')
        cls.settings = yaml.safe_load((ROOT/'emmoformer/config.yaml').read_text(encoding='utf-8'))

    def test_absolute_start_indices_reach_model_without_modulo(self):
        class Recorder:
            def __call__(self, x, starts):
                return starts
        starts = torch.tensor([6, 11, 12, 24, 115, 137])
        torch.testing.assert_close(forward(Recorder(), None, starts, 'EMMoFormer'), starts)

    def test_case_parameters_and_dimensions_follow_current_data(self):
        data, frame, base = prepare_data(self.global_cfg, 'zhejiang', 'selected')
        parameters, params = case_settings(self.settings, 'zhejiang', 'selected', epochs=1)
        kwargs = derive_model_kwargs(data, frame, base, parameters)
        self.assertEqual(params['epochs'], 1)
        self.assertEqual(self.settings['experiments']['zhejiang']['selected']['training']['epochs'], 160)
        self.assertEqual(kwargs['input_size'], len(data.feature_names))
        self.assertEqual(kwargs['target_index'], list(data.feature_names).index(TARGET))
        self.assertEqual((kwargs['seq_len'], kwargs['pred_len']), (6, 3))
        changed = frame.copy()
        changed.index = changed.index+1
        self.assertEqual(derive_model_kwargs(data, changed, base, parameters)['first_month_phase'], 1)
        for name in DERIVED_PARAMETERS:
            with self.subTest(parameter=name), self.assertRaises(ValueError):
                derive_model_kwargs(data, frame, base, dict(parameters, **{name: 1}))
        changed_base = copy.deepcopy(base)
        changed_base['y_scaler']['mean'][0] += 1
        with self.assertRaises(ValueError):
            derive_model_kwargs(data, frame, changed_base, parameters)

    def test_mape_loss_uses_actual_price_scale(self):
        actual = torch.tensor([[[0.0], [1.0], [-1.0]]])
        predicted = actual+0.1
        scaler = dict(mean=[4.0], scale=[2.0])
        expected = (0.2/torch.tensor([4.0, 6.0, 2.0])).mean()
        torch.testing.assert_close(loss_value(predicted, actual, {'loss': 'mape'}, scaler), expected)
        with self.assertRaises(ValueError):
            loss_value(predicted, actual-3, {'loss': 'mape'}, scaler)

    def test_six_final_checkpoints_match_configuration_and_preprocessing(self):
        for region in ('shanghai', 'zhejiang', 'jiangsu'):
            for profile in ('all', 'selected'):
                with self.subTest(region=region, profile=profile):
                    data, frame, base = prepare_data(self.global_cfg, region, profile)
                    parameters, params = case_settings(self.settings, region, profile)
                    model, saved = load_checkpoint(ROOT/'emmoformer/weights'/region/(profile+'.pt'),
                                                    'EMMoFormer', 'cpu')
                    self.assertEqual(saved['config']['model_kwargs'],
                                     derive_model_kwargs(data, frame, base, parameters))
                    self.assertEqual(saved['config']['training'], params)
                    self.assertEqual(saved['config']['features'], base['features'])
                    self.assertEqual(saved['config']['x_scaler'], base['x_scaler'])
                    self.assertEqual(saved['config']['y_scaler'], base['y_scaler'])
                    x, _ = data.test[0]
                    with torch.inference_mode():
                        prediction = forward(model, torch.from_numpy(x)[None], torch.tensor([115]), 'EMMoFormer')
                    self.assertEqual(tuple(prediction.shape), (1, 3, 1))
                    self.assertTrue(torch.isfinite(prediction).all())

    def test_cpu_one_epoch_training_reload_and_future_prediction(self):
        runtime = dict(self.global_cfg['runtime'], device='cpu')
        with tempfile.TemporaryDirectory(prefix='lng_emmoformer_') as temporary:
            output = Path(temporary)/'training'
            future = Path(temporary)/'future'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = train_one(self.global_cfg, self.settings, runtime, 'shanghai', 'selected',
                                   output, epochs=1)
                model, saved = load_checkpoint(output/'best.pt', 'EMMoFormer', 'cpu')
                future_result = predict_one(output/'best.pt', self.global_cfg, 'EMMoFormer',
                                            'cpu', future, mode='future')
            self.assertTrue(np.isfinite(result['MAPE_percent']))
            self.assertEqual(saved['best_epoch'], 1)
            self.assertEqual(saved['training_audit']['epochs_run'], 1)
            self.assertEqual(saved['config']['model_kwargs']['seq_len'], 6)
            self.assertEqual(len(pd.read_csv(output/'history.csv')), 1)
            preprocessing = json.loads((output/'preprocessing.json').read_text(encoding='utf-8'))
            self.assertEqual(preprocessing['x_scaler'], saved['config']['x_scaler'])
            table = pd.read_csv(future/'future_predictions.csv')
            self.assertEqual(len(table), 3)
            self.assertEqual(table['预测月份'].tolist(), ['2026-06', '2026-07', '2026-08'])
            self.assertTrue(np.isfinite(table['预测价格']).all())
            self.assertFalse(future_result['available'])


if __name__ == '__main__':
    unittest.main()
