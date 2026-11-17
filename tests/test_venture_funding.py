"""Model-validation regression tests on synthetic, non-sensitive applicants."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from venture_funding import TARGET, evaluate_models, load_applicants


class ModelTests(unittest.TestCase):
    def make_data(self, count=120):
        return pd.DataFrame({
            "EIN": [f"fake-{i}" for i in range(count)],
            "NAME": [f"Test Applicant {i}" for i in range(count)],
            "APPLICATION_TYPE": ["A" if i % 3 else "B" for i in range(count)],
            "AFFILIATION": ["Independent" if i % 2 else "Company" for i in range(count)],
            "INCOME_AMT": ["low" if i % 2 else "high" for i in range(count)],
            "ASK_AMT": [1000 + (i % 8) * 250 for i in range(count)],
            TARGET: [int(i % 2 == 0) for i in range(count)],
        })

    def write_data(self, frame):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "applicants.csv"
        frame.to_csv(path, index=False)
        return path

    def test_strips_identifiers_and_target_from_features(self):
        report = evaluate_models(self.make_data(), models=("baseline",))
        self.assertNotIn("EIN", report["input_features"])
        self.assertNotIn("NAME", report["input_features"])
        self.assertNotIn(TARGET, report["input_features"])
        self.assertEqual(report["train_count"] + report["test_count"], 120)

    def test_reproducible_logistic_and_baseline(self):
        frame = self.make_data()
        first = evaluate_models(frame, models=("baseline", "logistic"))
        second = evaluate_models(frame, models=("baseline", "logistic"))
        self.assertEqual(first, second)
        for model in first["models"].values():
            for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
                self.assertGreaterEqual(model[key], 0)
                self.assertLessEqual(model[key], 1)

    def test_mlp_trains_on_synthetic_data(self):
        report = evaluate_models(self.make_data(), models=("mlp",))
        self.assertIn("mlp", report["models"])

    def test_rejects_invalid_labels(self):
        for values in ([0] * 120, [2] * 120, [None] * 120):
            frame = self.make_data()
            frame[TARGET] = values
            with self.subTest(label=values[0]), self.assertRaises(ValueError):
                load_applicants(self.write_data(frame))

    def test_rejects_missing_target(self):
        with self.assertRaisesRegex(ValueError, TARGET):
            load_applicants(self.write_data(self.make_data().drop(columns=TARGET)))

    def test_rejects_invalid_test_sizes(self):
        for size in (0, 1, -0.2):
            with self.subTest(size=size), self.assertRaises(ValueError):
                evaluate_models(self.make_data(), test_size=size)

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.csv"
            report = Path(tmp) / "report.json"
            self.make_data().to_csv(data, index=False)
            subprocess.run(
                [sys.executable, "venture_funding.py", "--input", str(data),
                 "--output", str(report), "--model", "logistic"],
                cwd=Path(__file__).resolve().parents[1],
                check=True, capture_output=True, text=True,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["records"], 120)
            self.assertIn("logistic", payload["models"])


if __name__ == "__main__":
    unittest.main()
