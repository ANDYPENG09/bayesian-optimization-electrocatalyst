from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

CASE = Path(__file__).resolve().parents[1] / "examples" / "irox_screening"
sys.path.insert(0, str(CASE))

from benchmark import benchmark, design_matrices, write_report
from measurements import demo_inputs, validate_inputs


class IrOxScreeningTests(unittest.TestCase):
    def test_duplicate_scan_and_reopened_session_are_rejected(self):
        designs, scans = demo_inputs()
        duplicated = pd.concat([scans, scans.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "scan_order"):
            validate_inputs(designs, duplicated)

        batch = scans.iloc[1]["batch_id"]
        reopened = scans.copy()
        reopened.loc[reopened["batch_id"].eq(batch), "session_id"] = ["A", "B", "A"]
        with self.assertRaisesRegex(ValueError, "consecutive scans"):
            validate_inputs(designs, reopened)

    def test_ratio_columns_are_family_specific_and_fit_on_train(self):
        designs, scans = validate_inputs(*demo_inputs())
        a = designs.loc[designs["pore_family"].eq("NaHCO3")].iloc[:2].copy()
        b = designs.loc[designs["pore_family"].eq("NH3+NaCl+KCl")].iloc[:2].copy()
        a["pore_ratio"], b["pore_ratio"] = [2.0, 6.0], [50.0, 150.0]
        train = pd.concat([a, b], ignore_index=True)
        held_out = pd.concat([a.iloc[:1], b.iloc[:1]], ignore_index=True)
        held_out["pore_ratio"] = [8.0, np.nan]
        families = sorted(train["pore_family"].unique())
        x, x_test = design_matrices(train, held_out, sorted(train["precursor"].unique()), families)
        bicarbonate = families.index("NaHCO3")
        np.testing.assert_allclose(x[:2, -2 + bicarbonate], [-1.0, 1.0])
        np.testing.assert_allclose(x[2:, -2 + bicarbonate], [0.0, 0.0])
        np.testing.assert_allclose(x_test[0, -2 + bicarbonate], 2.0)
        np.testing.assert_allclose(x_test[1, -2:], [0.0, 0.0])

    def test_demo_benchmark_has_one_prediction_per_synthesis_per_model(self):
        designs, scans = demo_inputs()
        report, outputs = benchmark(designs, scans, "synthetic_demo")
        self.assertEqual(report["dataset_kind"], "synthetic_demo")
        self.assertEqual(report["counts"]["synthesis_batches"], 16)
        expected = set(designs.loc[designs["role"].eq("synthesis"), "batch_id"])
        predictions = outputs["loo_predictions"]
        full = predictions.loc[predictions["scope"].eq("historical")]
        for _, group in full.groupby(["target", "model"]):
            self.assertEqual(set(group["batch_id"]), expected)
            self.assertTrue(group["batch_id"].is_unique)
            self.assertTrue(np.isfinite(group["predicted_mv"]).all())
        self.assertEqual(len(full), 4 * len(expected))
        with tempfile.TemporaryDirectory() as directory:
            write_report(report, outputs, directory)
            self.assertTrue((Path(directory) / "summary.json").is_file())
