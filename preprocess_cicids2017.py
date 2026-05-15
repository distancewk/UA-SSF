"""
CICIDS-2017 preprocessing for SSF/UA-SSF experiments.

The script keeps the raw CSV untouched and writes reproducible binary
classification splits for the training script:

    CICIDS-2017/CICIDSTrain.csv
    CICIDS-2017/CICIDSTest.csv
    CICIDS-2017/CICIDSMetadata.csv
    CICIDS-2017/CICIDSManifest.json

Feature clipping and MinMax scaling are fitted on the training split only.
"""

from dataclasses import dataclass
import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class CicidsPreprocessResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    metadata_df: pd.DataFrame
    manifest: Dict


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return value


def _value_counts(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _binary_label_counts(series: pd.Series) -> Dict[str, int]:
    counts = series.astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def _prepare_numeric_frame(
        raw_df: pd.DataFrame,
        attack_column: str,
        normal_label: str) -> Tuple[pd.DataFrame, List[str]]:
    if attack_column not in raw_df.columns:
        raise ValueError(f"Missing attack label column: {attack_column}")

    feature_cols = [col for col in raw_df.columns if col != attack_column]
    if not feature_cols:
        raise ValueError("CICIDS input must contain at least one feature column")

    features = raw_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    prepared = features.copy()
    prepared["label"] = (raw_df[attack_column].astype(str) != normal_label).astype(int)
    prepared["sample_id"] = raw_df.index.astype(int)
    prepared["attack_type"] = raw_df[attack_column].astype(str).values
    return prepared, feature_cols


def _stratified_sample(df: pd.DataFrame, total_samples: int, seed: int) -> pd.DataFrame:
    if total_samples <= 0 or total_samples >= len(df):
        return df.copy()
    sampled, _ = train_test_split(
        df,
        train_size=total_samples,
        stratify=df["label"],
        random_state=seed,
    )
    return sampled.copy()


def scale_splits_with_train_statistics(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Iterable[str],
        clip_percentile: float = 99.9) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Clip and scale train/test using statistics fitted on the train split."""
    feature_cols = list(feature_cols)
    if not 50.0 <= clip_percentile <= 100.0:
        raise ValueError("clip_percentile must be in [50, 100]")

    train_features = train_df[feature_cols].astype(float)
    test_features = test_df[feature_cols].astype(float)

    lower_q = (100.0 - clip_percentile) / 100.0
    upper_q = clip_percentile / 100.0
    lower = train_features.quantile(lower_q)
    upper = train_features.quantile(upper_q)

    train_clipped = train_features.clip(lower=lower, upper=upper, axis=1)
    test_clipped = test_features.clip(lower=lower, upper=upper, axis=1)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_clipped)
    test_scaled = scaler.transform(test_clipped)

    scaled_train = pd.DataFrame(train_scaled, columns=feature_cols, index=train_df.index)
    scaled_test = pd.DataFrame(test_scaled, columns=feature_cols, index=test_df.index)
    scaled_train["label"] = train_df["label"].astype(int).values
    scaled_test["label"] = test_df["label"].astype(int).values

    scaler_stats = {
        "feature_count": len(feature_cols),
        "clip_percentile": float(clip_percentile),
        "clip_lower": lower.astype(float).to_dict(),
        "clip_upper": upper.astype(float).to_dict(),
        "train_min": pd.Series(scaler.data_min_, index=feature_cols).astype(float).to_dict(),
        "train_max": pd.Series(scaler.data_max_, index=feature_cols).astype(float).to_dict(),
    }
    return scaled_train.reset_index(drop=True), scaled_test.reset_index(drop=True), scaler_stats


def preprocess_cicids_dataframe(
        raw_df: pd.DataFrame,
        total_samples: int = 500000,
        test_ratio: float = 0.3,
        seed: int = 42,
        clip_percentile: float = 99.9,
        attack_column: str = "Attack Type",
        normal_label: str = "Normal Traffic") -> CicidsPreprocessResult:
    prepared, feature_cols = _prepare_numeric_frame(raw_df, attack_column, normal_label)
    sampled = _stratified_sample(prepared, total_samples, seed)

    train_raw, test_raw = train_test_split(
        sampled,
        test_size=test_ratio,
        stratify=sampled["label"],
        random_state=seed,
    )

    train_features = train_raw[feature_cols + ["label"]].copy()
    test_features = test_raw[feature_cols + ["label"]].copy()
    train_df, test_df, scaler_stats = scale_splits_with_train_statistics(
        train_features,
        test_features,
        feature_cols=feature_cols,
        clip_percentile=clip_percentile,
    )

    train_meta = train_raw[["sample_id", "attack_type", "label"]].copy()
    train_meta["split"] = "train"
    test_meta = test_raw[["sample_id", "attack_type", "label"]].copy()
    test_meta["split"] = "test"
    metadata_df = pd.concat([train_meta, test_meta], ignore_index=True)
    metadata_df = metadata_df[["sample_id", "split", "attack_type", "label"]]
    metadata_df["sample_id"] = metadata_df["sample_id"].astype(int)
    metadata_df["label"] = metadata_df["label"].astype(int)

    manifest = {
        "dataset": "CICIDS-2017",
        "raw_total_rows": int(len(raw_df)),
        "sampled_total_rows": int(len(sampled)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "total_samples": int(total_samples),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "clip_percentile": float(clip_percentile),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "label_mapping": {
            normal_label: 0,
            "Attack": 1,
        },
        "raw_attack_type_counts": _value_counts(raw_df[attack_column].astype(str)),
        "sampled_attack_type_counts": _value_counts(sampled["attack_type"]),
        "train_attack_type_counts": _value_counts(train_meta["attack_type"]),
        "test_attack_type_counts": _value_counts(test_meta["attack_type"]),
        "sampled_binary_label_counts": _binary_label_counts(sampled["label"]),
        "train_binary_label_counts": _binary_label_counts(train_raw["label"]),
        "test_binary_label_counts": _binary_label_counts(test_raw["label"]),
        "scaler": scaler_stats,
    }

    return CicidsPreprocessResult(
        train_df=train_df,
        test_df=test_df,
        metadata_df=metadata_df,
        manifest=_json_safe(manifest),
    )


def write_cicids_outputs(
        result: CicidsPreprocessResult,
        output_dir: str,
        train_filename: str = "CICIDSTrain.csv",
        test_filename: str = "CICIDSTest.csv",
        metadata_filename: str = "CICIDSMetadata.csv",
        manifest_filename: str = "CICIDSManifest.json") -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "train": os.path.join(output_dir, train_filename),
        "test": os.path.join(output_dir, test_filename),
        "metadata": os.path.join(output_dir, metadata_filename),
        "manifest": os.path.join(output_dir, manifest_filename),
    }

    result.train_df.to_csv(paths["train"], index=False)
    result.test_df.to_csv(paths["test"], index=False)
    result.metadata_df.to_csv(paths["metadata"], index=False)
    with open(paths["manifest"], "w", encoding="utf-8") as manifest_file:
        json.dump(_json_safe(result.manifest), manifest_file, ensure_ascii=False, indent=2)

    return paths


def _print_manifest_summary(manifest: Dict, paths: Dict[str, str]) -> None:
    print("Preprocessing complete")
    print(f"  Train: {paths['train']} ({manifest['train_rows']} rows)")
    print(f"  Test: {paths['test']} ({manifest['test_rows']} rows)")
    print(f"  Metadata: {paths['metadata']}")
    print(f"  Manifest: {paths['manifest']}")
    print(f"  Feature dimension: {manifest['feature_count']}")
    print(f"  Train labels: {manifest['train_binary_label_counts']}")
    print(f"  Test labels: {manifest['test_binary_label_counts']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CICIDS-2017 for UA-SSF")
    parser.add_argument("--input", type=str, default="CICIDS-2017/cicids2017_cleaned.csv",
                        help="Path to raw CICIDS-2017 CSV")
    parser.add_argument("--output_dir", type=str, default="CICIDS-2017",
                        help="Output directory for processed files")
    parser.add_argument("--total_samples", type=int, default=500000,
                        help="Total samples after stratified sampling; 0 uses all rows")
    parser.add_argument("--test_ratio", type=float, default=0.3,
                        help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling and split")
    parser.add_argument("--clip_percentile", type=float, default=99.9,
                        help="Upper percentile for train-fitted clipping")
    parser.add_argument("--metadata_filename", type=str, default="CICIDSMetadata.csv")
    parser.add_argument("--manifest_filename", type=str, default="CICIDSManifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading data from {args.input} ...")
    raw_df = pd.read_csv(args.input)
    print(f"  Raw shape: {raw_df.shape}")

    result = preprocess_cicids_dataframe(
        raw_df,
        total_samples=args.total_samples,
        test_ratio=args.test_ratio,
        seed=args.seed,
        clip_percentile=args.clip_percentile,
    )
    paths = write_cicids_outputs(
        result,
        output_dir=args.output_dir,
        metadata_filename=args.metadata_filename,
        manifest_filename=args.manifest_filename,
    )
    _print_manifest_summary(result.manifest, paths)


if __name__ == "__main__":
    main()
