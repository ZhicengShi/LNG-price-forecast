"""年度标签成熟边界和历史数据隔离检查。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annual_runtime import (VERSION, checkpoint_path, fill_inputs, load_checkpoint,
                            prepare_training, save_checkpoint)


class AnnualRuntimeTests(unittest.TestCase):
    def test_coregionlng_checkpoint_identifies_requested_model(self):
        spec = dict(model="CoRegionLng", profile="selected", regions=["shanghai", "zhejiang", "jiangsu"])
        path = checkpoint_path(Path(__file__).resolve().parents[1] / "weights", 2026, spec)
        loaded = load_checkpoint(path, 2026)
        self.assertEqual(loaded["spec"]["model"], spec["model"])
        self.assertEqual(loaded["spec"]["profile"], spec["profile"])

    def test_mature_labels_and_sample_count(self):
        months = pd.period_range("2015-01", "2024-08", freq="M")
        values = np.arange(1, len(months) + 1, dtype=float)
        raw = pd.DataFrame({"price": values}, index=months)
        spec = dict(seq_len=6, missing="ffill")
        x, positions, y, origins, scaler = prepare_training(raw, values[:, None], spec, "cpu")
        self.assertEqual(len(origins), 95)
        self.assertEqual(str(months[origins[-1]]), "2023-04")
        self.assertEqual(origins[-1] + 16, len(raw) - 1)
        self.assertEqual(tuple(x.shape), (95, 6, 1))
        self.assertEqual(tuple(y.shape), (95, 1, 1))
        np.testing.assert_array_equal(positions.numpy(), np.asarray(origins) - 5)
        actual = y.numpy()[:, 0, 0] * scaler["yscale"][0] + scaler["ymean"][0]
        expected = [values[t + 5:t + 17].mean() for t in origins]
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_filling_uses_only_supplied_history(self):
        raw = pd.DataFrame({"value": [1.0, np.nan, np.nan, 13.0]})
        limited = fill_inputs(raw.iloc[:3], [2.0], "linear_history")
        np.testing.assert_array_equal(limited[:, 0], [1.0, 1.0, 1.0])
        complete = fill_inputs(raw, [2.0], "linear_history")
        np.testing.assert_array_equal(complete[:, 0], [1.0, 5.0, 9.0, 13.0])

    def test_multichannel_windows_keep_contiguous_layout(self):
        raw = pd.DataFrame({f"x{i}": np.arange(30, dtype=float) + i for i in range(5)})
        x, _, _, _, _ = prepare_training(raw, raw[["x0"]].to_numpy(),
                                         dict(seq_len=6, missing="ffill"), "cpu")
        self.assertTrue(x.is_contiguous())
        self.assertEqual(x.stride(), (30, 5, 1))

    def test_incomplete_label_rejected(self):
        raw = pd.DataFrame({"price": np.arange(21, dtype=float)})
        with self.assertRaisesRegex(ValueError, "完整年度标签"):
            prepare_training(raw, raw.to_numpy(), dict(seq_len=6, missing="ffill"), "cpu")

    def test_weight_year_and_cutoff_checked(self):
        checkpoint = dict(version=VERSION, budget_year=2026, cutoff_month="2025-08",
                          state_dict={"sample": torch.ones(1)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            save_checkpoint(path, checkpoint)
            self.assertEqual(load_checkpoint(path, 2026)["budget_year"], 2026)
            with self.assertRaisesRegex(ValueError, "不匹配"):
                load_checkpoint(path, 2025)


if __name__ == "__main__":
    unittest.main()
