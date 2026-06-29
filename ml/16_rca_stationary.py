#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

import tensorflow as tf
from tensorflow import keras

sys.path.append(str(Path(__file__).parent))
import feature_utils

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"
FIRMWARE_DIR = ML_DIR.parent / "firmware"

def clean_csv(file_path):
    valid_rows = []
    with open(file_path, 'r', encoding='utf-16', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 6:
                try:
                    row = [float(p) for p in parts]
                    if any(np.isnan(v) for v in row):
                        continue
                    valid_rows.append(row)
                except ValueError:
                    continue
    return np.array(valid_rows, dtype=np.float64)

def main():
    print("="*60)
    print("ROOT CAUSE ANALYSIS: STATIONARY V3")
    print("="*60)
    
    # 1. Load stationary_v3
    v3_path = FIRMWARE_DIR / "stationary_v3.csv"
    v3_data = clean_csv(v3_path)
    print(f"Loaded stationary_v3.csv: {len(v3_data)} samples")
    
    # 2. Extract features
    dummy_labels = np.zeros(len(v3_data), dtype=np.int32)
    X_v3, _ = feature_utils.sliding_window_features(v3_data, dummy_labels)
    print(f"Extracted {len(X_v3)} windows from v3")
    
    # 3. Load V1 and V2
    s1_data = np.load(DATA_DIR / "session1_features.npz")
    X_s1 = s1_data['X'][s1_data['y'] == 0] # stationary only
    
    s2_data = np.load(DATA_DIR / "session2_features.npz")
    X_s2 = s2_data['X'][s2_data['y'] == 0]
    
    print(f"Loaded stationary_v1: {len(X_s1)} windows")
    print(f"Loaded stationary_v2: {len(X_s2)} windows")
    
    # 4. Feature Drift Comparison (first 6 features: mean of ax, ay, az, gx, gy, gz)
    print("\n--- Feature Mean Comparison (Top 6 Features: Means) ---")
    feat_names = ["mean_ax", "mean_ay", "mean_az", "mean_gx", "mean_gy", "mean_gz"]
    print(f"{'Feature':>10} | {'V1 (Train)':>12} | {'V2 (Test)':>12} | {'V3 (New)':>12}")
    for i in range(6):
        m1 = np.mean(X_s1[:, i])
        m2 = np.mean(X_s2[:, i])
        m3 = np.mean(X_v3[:, i])
        print(f"{feat_names[i]:>10} | {m1:12.4f} | {m2:12.4f} | {m3:12.4f}")
        
    print("\n--- Feature Variance Comparison (Next 6 Features: Vars) ---")
    feat_names = ["var_ax", "var_ay", "var_az", "var_gx", "var_gy", "var_gz"]
    for i in range(6):
        idx = i + 6
        v1 = np.mean(X_s1[:, idx])
        v2 = np.mean(X_s2[:, idx])
        v3 = np.mean(X_v3[:, idx])
        print(f"{feat_names[i]:>10} | {v1:12.4f} | {v2:12.4f} | {v3:12.4f}")

    # 5. Load Scaler and Model
    with open(MODELS_DIR / "real_scaler_params.json", 'r') as f:
        scaler_params = json.load(f)
        
    scale_mean = np.array(scaler_params['mean'])
    scale_scale = np.array(scaler_params['scale'])
    
    X_v3_scaled = (X_v3 - scale_mean) / scale_scale
    
    model = keras.models.load_model(MODELS_DIR / "real_fault_classifier.keras")
    
    # 6. Predict
    preds = model.predict(X_v3_scaled, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    
    counts = np.bincount(pred_classes, minlength=4)
    print("\n--- Python Model Predictions on V3 ---")
    labels = ["stationary", "movement", "rotation", "shake"]
    for i in range(4):
        print(f"{labels[i]:>10}: {counts[i]} windows ({(counts[i]/len(X_v3))*100:.1f}%)")
        
    if counts[1] > 0:
        mov_conf = np.mean(preds[pred_classes == 1, 1])
        print(f"Average confidence when predicting movement: {mov_conf*100:.1f}%")
        
    if counts[0] > 0:
        stat_conf = np.mean(preds[pred_classes == 0, 0])
        print(f"Average confidence when predicting stationary: {stat_conf*100:.1f}%")

if __name__ == "__main__":
    main()
