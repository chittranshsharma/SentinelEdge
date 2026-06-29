#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import sys

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"

def main():
    s1_data = np.load(DATA_DIR / "session1_features.npz")
    X_s1 = s1_data['X'][s1_data['y'] == 0]
    
    s2_data = np.load(DATA_DIR / "session2_features.npz")
    X_s2 = s2_data['X'][s2_data['y'] == 0]
    
    # Reload V3 using feature_utils
    sys.path.append(str(ML_DIR))
    import feature_utils
    v3_path = ML_DIR.parent / "firmware" / "stationary_v3.csv"
    valid_rows = []
    with open(v3_path, 'r', encoding='utf-16', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 6:
                try:
                    row = [float(p) for p in parts]
                    if not any(np.isnan(v) for v in row):
                        valid_rows.append(row)
                except ValueError:
                    pass
    v3_data = np.array(valid_rows, dtype=np.float64)
    dummy = np.zeros(len(v3_data), dtype=np.int32)
    X_v3, _ = feature_utils.sliding_window_features(v3_data, dummy)

    feat_indices = [0, 7, 14, 21, 28, 35]
    feat_names = ["ax_mean", "ay_mean", "az_mean", "gx_mean", "gy_mean", "gz_mean"]
    
    print(f"{'Feature':>10} | {'V1 (Train)':>12} | {'V2 (Test)':>12} | {'V3 (New)':>12}")
    for i, idx in enumerate(feat_indices):
        m1 = np.mean(X_s1[:, idx])
        m2 = np.mean(X_s2[:, idx])
        m3 = np.mean(X_v3[:, idx])
        print(f"{feat_names[i]:>10} | {m1:12.4f} | {m2:12.4f} | {m3:12.4f}")

if __name__ == "__main__":
    main()
