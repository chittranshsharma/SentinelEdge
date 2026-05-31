#!/usr/bin/env python3
"""
SentinelEdge — Feature Engineering
====================================
Applies sliding window feature extraction to the raw vibration CSV.
Imports the canonical feature extractor from feature_utils.py.

This script should NOT define feature math — it delegates to feature_utils.py.
This ensures the training pipeline and drift validation use identical logic.

Window configuration:
  WINDOW_SIZE = 200 samples  (2 seconds at 100Hz)
  STEP_SIZE   = 100 samples  (50% overlap → new prediction every 1 second)

Expected windows per class (6000 samples, 200-size, 100-step):
  (6000 - 200) // 100 + 1 = 59 windows per class → 236 windows total

Output: ml/data/features.npz
  X             : float64 array (n_windows, 42)
  y             : int32 array   (n_windows,)
  feature_names : str array     (42,)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

from feature_utils import (
    WINDOW_SIZE, STEP_SIZE, AXES, FEATURE_COUNT,
    build_feature_names, sliding_window_features,
    extract_features_window,
)

CLASSES = {0: "normal", 1: "imbalance", 2: "obstruction", 3: "loose_mount"}


def main():
    data_dir    = Path(__file__).parent / "data"
    input_path  = data_dir / "synthetic_vibration.csv"
    output_path = data_dir / "features.npz"

    print("=" * 62)
    print("SentinelEdge — Feature Engineering")
    print("=" * 62)

    if not input_path.exists():
        raise FileNotFoundError(
            f"\nInput CSV not found: {input_path}"
            f"\nRun 01_generate_synthetic_data.py first."
        )

    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Classes : {df['label'].value_counts().sort_index().to_dict()}")

    print(f"\nWindow configuration:")
    print(f"  Window size     : {WINDOW_SIZE} samples  "
          f"({WINDOW_SIZE/100:.1f}s at 100Hz)")
    print(f"  Step size       : {STEP_SIZE} samples  "
          f"(50% overlap → inference every {STEP_SIZE/100:.1f}s)")
    print(f"  Features/window : {FEATURE_COUNT}  ({len(AXES)} axes × 7 features)")

    # ── Extract features ───────────────────────────────────────────────────
    data   = df[AXES].values.astype(np.float64)
    labels = df['label'].values.astype(np.int32)

    print(f"\nExtracting features...")
    t0 = time.perf_counter()
    X, y = sliding_window_features(data, labels)
    elapsed = time.perf_counter() - t0

    feature_names = build_feature_names()

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype=str),
    )

    print(f"\n✓ Features saved: {output_path}")
    print(f"  Feature matrix shape : {X.shape}")
    print(f"  Label vector shape   : {y.shape}")
    print(f"  Extraction time      : {elapsed*1000:.1f}ms")
    print(f"  Time per window      : {elapsed/len(X)*1000:.3f}ms")

    print(f"\nClass distribution in windows:")
    for label, name in CLASSES.items():
        count = int(np.sum(y == label))
        print(f"  [{label}] {name:<16}: {count:>4} windows ({count/len(y)*100:.1f}%)")

    # ── Sanity checks ──────────────────────────────────────────────────────
    print(f"\nSanity checks:")

    # Check no NaN/Inf in features
    has_nan = np.any(np.isnan(X))
    has_inf = np.any(np.isinf(X))
    print(f"  NaN in features : {has_nan}  {'✗ FAIL' if has_nan else '✓ OK'}")
    print(f"  Inf in features : {has_inf}  {'✗ FAIL' if has_inf else '✓ OK'}")
    assert not has_nan, "NaN found in feature matrix!"
    assert not has_inf, "Inf found in feature matrix!"

    # Verify dominant_freq_bin is in valid range [1, WINDOW_SIZE//2]
    freq_bin_cols = [i * 7 + 5 for i in range(len(AXES))]  # indices of dominant_freq_bin features
    freq_bins = X[:, freq_bin_cols]
    print(f"  dominant_freq_bin range: [{freq_bins.min():.0f}, {freq_bins.max():.0f}]  "
          f"(valid: [1, {WINDOW_SIZE//2}])")
    assert freq_bins.min() >= 1, "dominant_freq_bin < 1 found!"
    assert freq_bins.max() <= WINDOW_SIZE // 2, "dominant_freq_bin > N/2 found!"

    # Verify class-specific dominant frequencies match expectations
    # Normal/Imbalance/Obstruction: ~25Hz → bin 50 (25 * 200/100 = 50)
    # Loose Mount: ~12Hz → bin 24 (12 * 200/100 = 24)
    print(f"\nDominant frequency verification (ax axis, bin → Hz):")
    for label, name in CLASSES.items():
        mask    = y == label
        ax_bins = X[mask, 5]  # ax_dominant_freq_bin (index 5)
        median_bin = np.median(ax_bins)
        median_hz  = median_bin * (100 / WINDOW_SIZE)
        print(f"  {name:<16}: median bin = {median_bin:5.1f}  "
              f"→  {median_hz:.1f} Hz  (expected: "
              f"{'25.0' if label != 3 else '12.0'} Hz)")

    # ── Feature statistics table ───────────────────────────────────────────
    print(f"\nFeature statistics (all 42 features):")
    print(f"  {'Feature':<32} {'Global Mean':>12} {'Global Std':>12} "
          f"{'Min':>12} {'Max':>12}")
    print(f"  {'─'*72}")
    for i, name in enumerate(feature_names):
        vals = X[:, i]
        print(f"  {name:<32} {vals.mean():>+12.4f} {vals.std():>12.4f} "
              f"{vals.min():>12.4f} {vals.max():>12.4f}")


if __name__ == "__main__":
    main()
