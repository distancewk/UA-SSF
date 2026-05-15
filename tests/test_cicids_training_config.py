import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CicidsTrainingConfigTests(unittest.TestCase):
    def test_cicids_splitdata_can_skip_second_normalization(self):
        source = (ROOT / "utils.py").read_text(encoding="utf-8")

        self.assertIn("def __init__(self, dataset, pre_scaled=False)", source)
        self.assertIn("self.pre_scaled = pre_scaled", source)
        self.assertIn("if self.pre_scaled:", source)
        self.assertIn("x_ = X_.to_numpy(dtype='float32')", source)

    def test_ssf_exposes_cicids_fair_experiment_controls(self):
        source = (ROOT / "ssf.py").read_text(encoding="utf-8")

        self.assertIn("--seed", source)
        self.assertIn("--seed_round", source)
        self.assertIn("--data_manifest", source)
        self.assertIn("--disable_uncertainty_weight", source)
        self.assertIn("--disable_selective_distillation", source)
        self.assertIn("--disable_adaptive_drift", source)
        self.assertIn("ablation_name", source)
        self.assertIn("[Summary]", source)


if __name__ == "__main__":
    unittest.main()
