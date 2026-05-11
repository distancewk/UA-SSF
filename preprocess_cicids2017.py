"""
CICIDS-2017 数据集预处理脚本
=============================
将原始 cicids2017_cleaned.csv 转换为与 UNSW 格式一致的 Train/Test CSV 文件，
以便直接接入 UA-SSF 持续学习框架。

处理步骤：
1. 读取原始 CSV (52 特征 + Attack Type 标签)
2. 二分类标签转换: Normal Traffic → 0, 攻击 → 1
3. 异常值裁剪 (clip to 99.9th percentile)
4. 分层采样 (~500K)
5. Train/Test 分割 (7:3)
6. 输出: CICIDS-2017/CICIDSTrain.csv, CICIDS-2017/CICIDSTest.csv

Usage:
    python preprocess_cicids2017.py [--total_samples 500000] [--test_ratio 0.3] [--seed 42]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='CICIDS-2017 Preprocessing for UA-SSF')
    parser.add_argument('--input', type=str, default='CICIDS-2017/cicids2017_cleaned.csv',
                        help='Path to raw CICIDS-2017 CSV')
    parser.add_argument('--output_dir', type=str, default='CICIDS-2017',
                        help='Output directory for processed files')
    parser.add_argument('--total_samples', type=int, default=500000,
                        help='Total samples after stratified sampling (0=use all)')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--clip_percentile', type=float, default=99.9,
                        help='Percentile for outlier clipping')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ========== Step 1: 读取原始数据 ==========
    print(f"[Step 1] Loading data from {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {len(df.columns)} (features: {len(df.columns) - 1}, label: 1)")

    # ========== Step 2: 二分类标签转换 ==========
    print(f"\n[Step 2] Converting labels to binary ...")
    print(f"  Original Attack Type distribution:")
    print(df['Attack Type'].value_counts().to_string(header=False))

    # Normal Traffic → 0, 所有攻击类型 → 1
    df['label'] = (df['Attack Type'] != 'Normal Traffic').astype(int)
    df = df.drop('Attack Type', axis=1)

    print(f"\n  Binary label distribution:")
    print(f"    Normal (0): {(df['label'] == 0).sum()}")
    print(f"    Attack (1): {(df['label'] == 1).sum()}")
    print(f"    Attack ratio: {df['label'].mean():.4f}")

    # ========== Step 3: 数据清洗 ==========
    print(f"\n[Step 3] Cleaning data ...")

    # 确保所有特征列为数值型
    feature_cols = [c for c in df.columns if c != 'label']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 处理 NaN (由 coerce 产生)
    nan_before = df.isnull().sum().sum()
    if nan_before > 0:
        print(f"  Found {nan_before} NaN values after numeric conversion, filling with 0")
        df = df.fillna(0)

    # 处理 Inf
    inf_count = np.isinf(df[feature_cols].values).sum()
    if inf_count > 0:
        print(f"  Found {inf_count} Inf values, replacing with column max")
        for col in feature_cols:
            col_data = df[col]
            finite_max = col_data[np.isfinite(col_data)].max()
            df[col] = col_data.replace([np.inf, -np.inf], [finite_max, -finite_max])

    # 异常值裁剪: clip to 99.9th percentile
    print(f"  Clipping outliers to {args.clip_percentile}th percentile ...")
    for col in feature_cols:
        upper = np.percentile(df[col].values, args.clip_percentile)
        lower = np.percentile(df[col].values, 100 - args.clip_percentile)
        df[col] = df[col].clip(lower=lower, upper=upper)

    print(f"  Cleaned shape: {df.shape}")

    # ========== Step 4: 分层采样 ==========
    if args.total_samples > 0 and args.total_samples < len(df):
        print(f"\n[Step 4] Stratified sampling to {args.total_samples} samples ...")
        sample_ratio = args.total_samples / len(df)

        df_sampled, _ = train_test_split(
            df, train_size=args.total_samples, stratify=df['label'],
            random_state=args.seed
        )
        df = df_sampled
        print(f"  Sampled shape: {df.shape}")
        print(f"  Normal (0): {(df['label'] == 0).sum()}")
        print(f"  Attack (1): {(df['label'] == 1).sum()}")
        print(f"  Attack ratio: {df['label'].mean():.4f}")
    else:
        print(f"\n[Step 4] Using all {len(df)} samples (no sampling)")

    # ========== Step 5: Train/Test 分割 ==========
    print(f"\n[Step 5] Splitting into train/test (ratio={args.test_ratio}) ...")

    train_df, test_df = train_test_split(
        df, test_size=args.test_ratio, stratify=df['label'],
        random_state=args.seed
    )

    print(f"  Train shape: {train_df.shape}")
    print(f"    Normal: {(train_df['label'] == 0).sum()}, Attack: {(train_df['label'] == 1).sum()}")
    print(f"  Test shape: {test_df.shape}")
    print(f"    Normal: {(test_df['label'] == 0).sum()}, Attack: {(test_df['label'] == 1).sum()}")

    # ========== Step 6: 保存 ==========
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, 'CICIDSTrain.csv')
    test_path = os.path.join(args.output_dir, 'CICIDSTest.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[Step 6] Saved:")
    print(f"  Train: {train_path} ({os.path.getsize(train_path) / 1024 / 1024:.1f} MB)")
    print(f"  Test:  {test_path} ({os.path.getsize(test_path) / 1024 / 1024:.1f} MB)")

    # ========== 最终统计 ==========
    feature_dim = len(feature_cols)
    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Total samples: {len(train_df) + len(test_df)}")
    print(f"  Train/Test: {len(train_df)} / {len(test_df)}")
    print(f"  → ssf.py 使用: --dataset cicids, input_dim={feature_dim}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
