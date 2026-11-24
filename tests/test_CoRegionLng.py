"""融合模型的布局、历史隔离、候选计划和区间边界检查。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CoRegionLng._runtime import read_config, frame_for
from CoRegionLng.data import fit_preprocessing, windows, REGIONS
from CoRegionLng.model import CoRegionLng
from CoRegionLng._search import candidates
from CoRegionLng.analysis import block_indices, confidence_rows, prediction_intervals
from CoRegionLng.backtest import completed_origins


class CoRegionLngTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_layout_matches_saved_weights_for_both_profiles(self):
        index = json.loads((ROOT/'CoRegionLng/reference/weights_index.json').read_text(encoding='utf-8'))
        for entry in index:
            cfg, shared, _, _ = read_config(ROOT/f'CoRegionLng/configs/{entry["profile"]}.yaml')
            frame, layout, versions = frame_for(shared, cfg['profile'], shared['period']['test_end'])
            self.assertEqual(layout, entry['input_layout'])
            self.assertEqual(set(versions), set(REGIONS))
            self.assertTrue(all(item['filename'].endswith('.xlsx') for item in versions.values()))
            self.assertEqual(frame.shape[0], 137)
            if cfg['profile'] == 'selected':
                self.assertEqual([len(c) for c in layout['branch_indices']], [4, 3, 5])

    def test_future_values_do_not_fill_past_inputs_or_fit_preprocessing(self):
        cfg, shared, _, _ = read_config(ROOT/'CoRegionLng/configs/selected.yaml')
        frame, layout, _ = frame_for(shared, 'selected', shared['period']['test_end'])
        frame.iloc[94, 1] = np.nan
        prep = fit_preprocessing(frame, layout, 90, 'linear_history')
        changed = frame.copy()
        changed.iloc[95:] = changed.iloc[95:]*100+1
        changed_prep = fit_preprocessing(changed, layout, 90, 'linear_history')
        self.assertEqual(prep, changed_prep)
        original = windows(frame, layout, prep, 95, 98, 6)
        altered = windows(changed, layout, prep, 95, 98, 6)
        self.assertTrue(torch.equal(original.x, altered.x))
        self.assertEqual(original.starts, [95])
        train = windows(frame, layout, prep, 0, 90, 6)
        self.assertEqual(train.starts[-1]+2, 89)
        frame.iloc[96, layout['target_indices'][0]] = np.nan
        with self.assertRaises(ValueError):
            windows(frame, layout, prep, 95, 98, 6)

    def test_delta_head_initial_output_is_last_price_and_encoder_is_shared(self):
        cfg, shared, _, _ = read_config(ROOT/'CoRegionLng/configs/selected.yaml')
        _, layout, _ = frame_for(shared, 'selected', shared['period']['test_end'])
        p = dict(cfg['parameters'], d_model=8, n_heads=2, e_layers=1, ff_multiplier=2,
                 head_layers=2, head_multiplier=1, cycle=0)
        model = CoRegionLng(layout, p).eval()
        x = torch.randn(2, 6, layout['input_size'])
        with torch.inference_mode():
            predicted = model(x, torch.tensor([6, 7]))
        expected = x[:, -1, layout['target_indices']][:, None, :].repeat(1, 3, 1)
        self.assertTrue(torch.equal(predicted, expected))
        self.assertTrue(all(not hasattr(branch, 'encoder') for branch in model.branches))
        self.assertTrue(hasattr(model, 'encoder'))

    def test_search_plan_matches_reference_64_candidates(self):
        cfg = yaml.safe_load((ROOT/'CoRegionLng/configs/search.yaml').read_text(encoding='utf-8'))
        expected = json.loads((ROOT/'CoRegionLng/reference/search/search_plan.json').read_text(encoding='utf-8'))
        self.assertEqual(candidates(cfg), expected)

    def test_score_interval_is_inverse_mape_interval(self):
        y = np.ones((30, 3, 3))
        forecast = y+np.arange(30)[:, None, None]/1000
        indexes = block_indices(30, 12, 100, 42)
        rows = confidence_rows(y, forecast, indexes, 'selected', 'backtest', 12)
        for region in REGIONS:
            for horizon in ['all', '1', '2', '3']:
                subset = {r['metric']: r for r in rows if r['region'] == region and r['horizon'] == horizon}
                self.assertAlmostEqual(subset['score']['lower'], 100-subset['MAPE']['upper'])
                self.assertAlmostEqual(subset['score']['upper'], 100-subset['MAPE']['lower'])

    def test_prediction_interval_uses_only_matured_residuals(self):
        table = pd.DataFrame([dict(profile='selected', region=r, horizon=h, origin_index=t,
            target_month=str(pd.Period('2015-01', freq='M')+t+h-1), actual=4., prediction=4.+t/100)
            for t in range(40) for h in [1, 2, 3] for r in REGIONS])
        before = prediction_intervals(table, 5, 20)
        table.loc[table.origin_index >= 30, 'actual'] = 10000
        after = prediction_intervals(table, 5, 20)
        for frame in (before, after):
            ready = frame.latest_calibration_target_index.notna()
            self.assertTrue((frame.loc[ready, 'latest_calibration_target_index'] < frame.loc[ready, 'origin_index']).all())
        np.testing.assert_array_equal(before[before.origin_index == 30][['lower', 'upper']],
                                      after[after.origin_index == 30][['lower', 'upper']])

    def test_resume_rejects_incomplete_or_duplicate_origin_rows(self):
        table = pd.DataFrame([dict(origin_index=9, horizon=h, region=r, actual=4., prediction=4.)
                              for h in [1, 2, 3] for r in REGIONS])
        audit = [dict(origin_index=9)]
        self.assertEqual(completed_origins(table, audit, [9]), {9})
        self.assertEqual(completed_origins(table.iloc[:-1], audit, [9]), set())
        table.loc[0, 'horizon'] = 2
        self.assertEqual(completed_origins(table, audit, [9]), set())


if __name__ == '__main__':
    unittest.main()
