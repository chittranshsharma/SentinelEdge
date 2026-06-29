#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import random

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def main():
    set_seeds()
    print("="*60)
    print("FEATURE ABLATION STUDY: Removing dominant_freq_bin")
    print("="*60)
    
    ML_DIR = Path(__file__).parent
    DATA_DIR = ML_DIR / "data"
    
    s1_data = np.load(DATA_DIR / "session1_features.npz")
    X_train_full = s1_data['X']
    y_train = s1_data['y']
    
    s2_data = np.load(DATA_DIR / "session2_features.npz")
    X_test_full = s2_data['X']
    y_test = s2_data['y']
    
    # Baseline accuracy (42 features)
    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_test_full_scaled = scaler_full.transform(X_test_full)
    
    lr_full = LogisticRegression(max_iter=1000)
    lr_full.fit(X_train_full_scaled, y_train)
    lr_acc_full = accuracy_score(y_test, lr_full.predict(X_test_full_scaled))
    
    # ── Remove dominant_freq_bin ─────────────────────────────────────────────
    # Features per axis: mean, std, variance, rms, peak_to_peak, dom_bin, energy
    # dom_bin is at index 5, 12, 19, 26, 33, 40
    dom_indices = [5, 12, 19, 26, 33, 40]
    keep_indices = [i for i in range(42) if i not in dom_indices]
    
    X_train_abl = X_train_full[:, keep_indices]
    X_test_abl = X_test_full[:, keep_indices]
    
    print(f"Original feature count: {X_train_full.shape[1]}")
    print(f"Ablated feature count : {X_train_abl.shape[1]}\n")
    
    scaler_abl = StandardScaler()
    X_train_abl_scaled = scaler_abl.fit_transform(X_train_abl)
    X_test_abl_scaled = scaler_abl.transform(X_test_abl)
    
    # Logistic Regression
    lr_abl = LogisticRegression(max_iter=1000)
    lr_abl.fit(X_train_abl_scaled, y_train)
    lr_acc_abl = accuracy_score(y_test, lr_abl.predict(X_test_abl_scaled))
    
    # Dense(4) Ablated
    model_abl = Sequential([
        Input(shape=(36,)),
        Dense(4, activation='softmax')
    ])
    model_abl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    
    model_abl.fit(X_train_abl_scaled, y_train, epochs=1000, batch_size=32, verbose=0)
    _, nn_acc_abl = model_abl.evaluate(X_test_abl_scaled, y_test, verbose=0)
    
    # Retrain Dense(4) baseline just in case to ensure fairness
    model_full = Sequential([
        Input(shape=(42,)),
        Dense(4, activation='softmax')
    ])
    model_full.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    model_full.fit(X_train_full_scaled, y_train, epochs=1000, batch_size=32, verbose=0)
    _, nn_acc_full = model_full.evaluate(X_test_full_scaled, y_test, verbose=0)
    
    print("--- Session Holdout Accuracy ---")
    print(f"{'Model':<25} | {'42 Features':<12} | {'36 Features (Ablated)':<25}")
    print("-" * 70)
    print(f"{'Logistic Regression':<25} | {lr_acc_full*100:10.2f}% | {lr_acc_abl*100:10.2f}%")
    print(f"{'Dense(4) Softmax':<25} | {nn_acc_full*100:10.2f}% | {nn_acc_abl*100:10.2f}%")
    
if __name__ == "__main__":
    main()
