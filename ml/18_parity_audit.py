#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent))
import feature_utils

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"
FIRMWARE_DIR = ML_DIR.parent / "firmware"

def cpp_extract_features(window):
    """
    Exactly simulates the C++ extractFeatures() logic in firmware/src/features.cpp
    """
    N = 200
    FFT_SIZE = 256
    features = np.zeros(42, dtype=np.float64)
    
    for axis in range(6):
        x = window[:, axis]
        
        # Time-domain pass
        mean = np.sum(x) / N
        
        sumSqDiff = 0.0
        sumSq = 0.0
        xMin = x[0]
        xMax = x[0]
        
        for i in range(N):
            v = x[i]
            diff = v - mean
            sumSqDiff += diff * diff
            sumSq += v * v
            if v < xMin: xMin = v
            if v > xMax: xMax = v
            
        variance = sumSqDiff / N
        std_val = np.sqrt(variance)
        rms = np.sqrt(sumSq / N)
        peak2peak = xMax - xMin
        
        # FFT pass
        # Zero-pad to 256
        x_padded = np.zeros(FFT_SIZE, dtype=np.float64)
        x_padded[:N] = x - mean
        
        # Compute exact DFT (matching arduinoFFT without windowing)
        # arduinoFFT complexToMagnitude calculates sqrt(real^2 + imag^2)
        fft_vals = np.fft.rfft(x_padded)
        magnitudes = np.abs(fft_vals)
        
        halfN = 100
        domBin = 1
        maxMag = magnitudes[1]
        specEnergy = 0.0
        
        for k in range(1, halfN + 1):
            mag = magnitudes[k]
            if mag > maxMag:
                maxMag = mag
                domBin = k
            specEnergy += mag * mag
            
        idx = axis * 7
        features[idx+0] = mean
        features[idx+1] = std_val
        features[idx+2] = variance
        features[idx+3] = rms
        features[idx+4] = peak2peak
        features[idx+5] = domBin
        features[idx+6] = specEnergy
        
    return features

def main():
    print("="*80)
    print("FEATURE PARITY AUDIT: Python vs C++ Simulation")
    print("="*80)
    
    # 1. Load stationary_v3 window 0
    v3_path = FIRMWARE_DIR / "stationary_v3.csv"
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
    window = X_v3[0] # Not the raw 200 samples, wait!
    # X_v3 is already features. I need the raw 200 samples.
    raw_window = v3_data[:200]
    
    # 2. Extract features
    py_features = feature_utils.extract_features_window(raw_window)
    cpp_features = cpp_extract_features(raw_window)
    
    # 3. Load Scaler
    with open(MODELS_DIR / "real_scaler_params.json", 'r') as f:
        scaler_params = json.load(f)
    scale_mean = np.array(scaler_params['mean'])
    scale_scale = np.array(scaler_params['scale'])
    
    # 4. Compare
    print(f"{'Feature Name':<25} | {'Python':>12} | {'C++':>12} | {'Delta':>12} | {'Z-Score Diff':>12}")
    print("-" * 80)
    
    feature_names = []
    axes = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    feats = ['mean', 'std', 'var', 'rms', 'p2p', 'dom_bin', 'energy']
    for a in axes:
        for f in feats:
            feature_names.append(f"{a}_{f}")
            
    drifts = []
    
    for i in range(42):
        py_v = py_features[i]
        cpp_v = cpp_features[i]
        delta = abs(py_v - cpp_v)
        z_diff = delta / scale_scale[i]
        drifts.append((z_diff, feature_names[i], py_v, cpp_v, delta))
        
    drifts.sort(key=lambda x: x[0], reverse=True)
    
    for drift in drifts:
        z_diff, name, py_v, cpp_v, delta = drift
        print(f"{name:<25} | {py_v:12.4f} | {cpp_v:12.4f} | {delta:12.4f} | {z_diff:12.4f}")

if __name__ == "__main__":
    main()
