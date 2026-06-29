import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import canonical feature extraction
sys.path.append(str(Path(__file__).parent))
import feature_utils

DATA_DIR = Path(__file__).parent / "data"
BACKUP_DIR = DATA_DIR / "raw_backup"

LABELS = {
    "stationary": 0,
    "movement": 1,
    "rotation": 2,
    "shake": 3
}

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
                    # Also check for NaN
                    if any(np.isnan(v) for v in row):
                        continue
                    valid_rows.append(row)
                except ValueError:
                    # Skip boot logs or non-numeric headers
                    continue
    return np.array(valid_rows, dtype=np.float64)

def main():
    print("PHASE 1: Cleaning and loading datasets...")
    
    all_data = []
    all_labels = []
    
    samples_per_class = {}
    
    for class_name, label in LABELS.items():
        file_path = BACKUP_DIR / f"{class_name}.csv"
        if not file_path.exists():
            print(f"Error: {file_path} not found.")
            sys.exit(1)
            
        data = clean_csv(file_path)
        print(f"  {class_name}.csv -> {len(data)} valid numeric samples")
        
        samples_per_class[class_name] = len(data)
        
        labels = np.full(len(data), label, dtype=np.int32)
        
        all_data.append(data)
        all_labels.append(labels)
        
    final_data = np.vstack(all_data)
    final_labels = np.concatenate(all_labels)
    
    print("\nTotal valid samples:", len(final_data))
    
    # Save combined raw data
    raw_output_path = DATA_DIR / "real_dataset.csv"
    print(f"\nSaving cleaned raw dataset to {raw_output_path} ...")
    
    df = pd.DataFrame(final_data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    df['label'] = final_labels
    df.to_csv(raw_output_path, index=False)
    
    print("\nPHASE 2 & 3: Chronological Splitting, Windowing, and Feature Extraction...")
    
    train_X_list, train_y_list = [], []
    test_X_list, test_y_list = [], []
    
    train_windows_per_class = {}
    test_windows_per_class = {}
    
    for class_name, label in LABELS.items():
        idx = label
        class_data = all_data[idx]
        class_labels = all_labels[idx]
        
        split_idx = int(len(class_data) * 0.8)
        
        train_data = class_data[:split_idx]
        train_labels = class_labels[:split_idx]
        
        test_data = class_data[split_idx:]
        test_labels = class_labels[split_idx:]
        
        # Window train and test independently
        X_train, y_train = feature_utils.sliding_window_features(train_data, train_labels)
        X_test, y_test = feature_utils.sliding_window_features(test_data, test_labels)
        
        train_X_list.append(X_train)
        train_y_list.append(y_train)
        train_windows_per_class[class_name] = len(X_train)
        
        test_X_list.append(X_test)
        test_y_list.append(y_test)
        test_windows_per_class[class_name] = len(X_test)
        
    final_X_train = np.vstack(train_X_list)
    final_y_train = np.concatenate(train_y_list)
    
    final_X_test = np.vstack(test_X_list)
    final_y_test = np.concatenate(test_y_list)
    
    print("\nSummary Statistics:")
    print("-------------------")
    for class_name in LABELS.keys():
        s = samples_per_class[class_name]
        tr_w = train_windows_per_class[class_name]
        te_w = test_windows_per_class[class_name]
        print(f"Class: {class_name:12s} | Samples: {s:6d} | Train Windows: {tr_w:5d} | Test Windows: {te_w:5d}")
        
    print(f"\nTotal Train Windows: {len(final_y_train)}")
    print(f"Total Test Windows: {len(final_y_test)}")
    
    train_out_path = DATA_DIR / "real_features_train.npz"
    test_out_path = DATA_DIR / "real_features_test.npz"
    
    np.savez_compressed(train_out_path, X=final_X_train, y=final_y_train)
    np.savez_compressed(test_out_path, X=final_X_test, y=final_y_test)
    print(f"\nSaved {train_out_path}")
    print(f"Saved {test_out_path}")

if __name__ == '__main__':
    main()
