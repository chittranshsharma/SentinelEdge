#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import canonical feature extraction
sys.path.append(str(Path(__file__).parent))
import feature_utils

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"
SESSION1_DIR = DATA_DIR / "raw_backup"
SESSION2_DIR = ML_DIR.parent / "firmware"

LABELS = {
    "stationary": 0,
    "movement": 1,
    "rotation": 2,
    "shake": 3
}

def clean_csv(file_path):
    valid_rows = []
    # Powershell Tee-Object outputs UTF-16
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

def process_session(session_dir, suffix, out_name):
    print(f"\nProcessing {out_name} from {session_dir} (suffix: {suffix})...")
    all_X = []
    all_y = []
    
    total_samples = 0
    total_windows = 0
    
    for class_name, label in LABELS.items():
        filename = f"{class_name}{suffix}.csv"
        file_path = session_dir / filename
        if not file_path.exists():
            print(f"Error: {file_path} not found.")
            sys.exit(1)
            
        data = clean_csv(file_path)
        samples = len(data)
        total_samples += samples
        
        labels = np.full(samples, label, dtype=np.int32)
        
        X, y = feature_utils.sliding_window_features(data, labels)
        windows = len(X)
        total_windows += windows
        
        print(f"  {class_name:12s} -> {samples:6d} samples, {windows:4d} windows")
        
        all_X.append(X)
        all_y.append(y)
        
    final_X = np.vstack(all_X)
    final_y = np.concatenate(all_y)
    
    out_path = DATA_DIR / f"{out_name}.npz"
    np.savez_compressed(out_path, X=final_X, y=final_y)
    
    print(f"Total {out_name}: {total_samples} samples -> {total_windows} windows")
    print(f"Saved {out_path}")

def main():
    print("PHASE 1: Session 1 (Train)")
    process_session(SESSION1_DIR, "", "session1_features")
    
    print("\nPHASE 2: Session 2 (Test)")
    process_session(SESSION2_DIR, "_v2", "session2_features")

if __name__ == '__main__':
    main()
