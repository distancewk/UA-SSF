import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess_cicids2017 import (
    preprocess_cicids_dataframe,
    scale_splits_with_train_statistics,
)


class CicidsPreprocessingTests(unittest.TestCase):
    def test_scale_splits_uses_training_statistics(self):
        train_df = pd.DataFrame({
            "feature_a": [0.0, 10.0, 20.0, 30.0],
            "feature_b": [5.0, 15.0, 25.0, 35.0],
            "label": [0, 0, 1, 1],
        })
        test_df = pd.DataFrame({
            "feature_a": [15.0, 15.0],
            "feature_b": [20.0, 20.0],
            "label": [0, 1],
        })

        scaled_train, scaled_test, scaler_stats = scale_splits_with_train_statistics(
            train_df,
            test_df,
            feature_cols=["feature_a", "feature_b"],
            clip_percentile=100.0,
        )

        np.testing.assert_allclose(scaled_train["feature_a"], [0.0, 1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(scaled_test["feature_a"], [0.5, 0.5])
        np.testing.assert_allclose(scaled_test["feature_b"], [0.5, 0.5])
        self.assertEqual(scaler_stats["feature_count"], 2)
        self.assertIn("feature_a", scaler_stats["train_min"])
        self.assertEqual(scaled_train["label"].tolist(), [0, 0, 1, 1])

    def test_preprocess_dataframe_writes_binary_features_metadata_and_manifest(self):
        rows = []
        labels = ["Normal Traffic"] * 18 + ["DoS"] * 9 + ["DDoS"] * 3
        for idx, attack_type in enumerate(labels):
            rows.append({
                "Destination Port": idx % 5,
                "Flow Duration": 100 + idx,
                "Attack Type": attack_type,
            })
        df = pd.DataFrame(rows)

        result = preprocess_cicids_dataframe(
            df,
            total_samples=20,
            test_ratio=0.3,
            seed=42,
            clip_percentile=100.0,
        )

        expected_columns = ["Destination Port", "Flow Duration", "label"]
        self.assertEqual(result.train_df.columns.tolist(), expected_columns)
        self.assertEqual(result.test_df.columns.tolist(), expected_columns)
        self.assertNotIn("Attack Type", result.train_df.columns)
        self.assertEqual(set(result.train_df["label"].unique()).union(result.test_df["label"].unique()), {0, 1})

        self.assertEqual(len(result.train_df), 14)
        self.assertEqual(len(result.test_df), 6)
        self.assertEqual(len(result.metadata_df), 20)
        self.assertEqual(result.metadata_df.columns.tolist(), ["sample_id", "split", "attack_type", "label"])
        self.assertEqual(set(result.metadata_df["split"]), {"train", "test"})
        self.assertIn("DoS", set(result.metadata_df["attack_type"]))

        manifest = result.manifest
        self.assertEqual(manifest["raw_total_rows"], 30)
        self.assertEqual(manifest["sampled_total_rows"], 20)
        self.assertEqual(manifest["train_rows"], 14)
        self.assertEqual(manifest["test_rows"], 6)
        self.assertEqual(manifest["feature_count"], 2)
        self.assertEqual(manifest["label_mapping"]["Normal Traffic"], 0)
        self.assertEqual(manifest["label_mapping"]["Attack"], 1)


if __name__ == "__main__":
    unittest.main()
