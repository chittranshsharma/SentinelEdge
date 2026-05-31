#!/usr/bin/env python3
"""
SentinelEdge — Model Training
================================
Trains fault detection models on extracted features.

V1 — Isolation Forest (anomaly detection, binary, trained on normal only)
     Used for initial system validation and threshold tuning.

V2 — Dense Neural Network (4-class fault classification)
     Architecture: Input(42) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(4, Softmax)
     Designed for TFLite int8 quantization on ESP32.

Performance targets (assertions will fail if not met):
  Accuracy      : > 85%
  FPR           : < 10%
  Total params  : < 2000  (fits easily in quantized TFLite)

Outputs:
  models/fault_classifier.keras   — Keras model
  models/isolation_forest.pkl     — sklearn model
  models/scaler_params.json       — StandardScaler mean + scale (needed by firmware)
  models/keras_metrics.json       — training metrics (consumed by 04_convert_to_tflite.py)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json, time, pickle, warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)

# ── Constants ──────────────────────────────────────────────────────────────────

CLASSES       = {0: "normal", 1: "imbalance", 2: "obstruction", 3: "loose_mount"}
NUM_CLASSES   = 4
FEATURE_COUNT = 42
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

# ── Model Architecture ─────────────────────────────────────────────────────────

def build_nn(input_dim: int = FEATURE_COUNT, num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Tiny dense network — optimized for TFLite int8 quantization.

    Architecture chosen for:
    - Parameter count < 2000 (→ model size < 20KB after int8 quant)
    - All Dense layers (fully supported by TFLite Micro)
    - No LSTM/Conv (limited TFLite Micro support on ESP32)

    Total trainable params: (42×32+32) + (32×16+16) + (16×4+4) = 1,924
    """
    inputs = keras.Input(shape=(input_dim,), name='features')
    x = keras.layers.Dense(32, activation='relu', name='dense_1')(inputs)
    x = keras.layers.Dense(16, activation='relu', name='dense_2')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='probabilities')(x)
    return keras.Model(inputs, outputs, name='SentinelEdge_FaultClassifier')


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_nn(model, X_test_scaled, y_test):
    """Run full evaluation and return metrics dict."""
    y_proba = model.predict(X_test_scaled, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    cm  = confusion_matrix(y_test, y_pred)

    # False positive rate: normal samples predicted as ANY fault class
    normal_mask = (y_test == 0)
    fpr = float(np.sum((y_pred != 0) & normal_mask) / normal_mask.sum())

    prec  = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec   = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1    = f1_score(y_test, y_pred, average=None, zero_division=0)

    return {
        'accuracy':            acc,
        'false_positive_rate': fpr,
        'precision_per_class': prec.tolist(),
        'recall_per_class':    rec.tolist(),
        'f1_per_class':        f1.tolist(),
        'confusion_matrix':    cm.tolist(),
    }


def print_metrics(metrics: dict, label: str = ""):
    if label:
        print(f"\n  [{label}]")
    print(f"  Accuracy            : {metrics['accuracy']*100:.2f}%")
    print(f"  False Positive Rate : {metrics['false_positive_rate']*100:.2f}%")

    print(f"\n  {'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*50}")
    for i, name in CLASSES.items():
        print(f"  {name:<18} "
              f"{metrics['precision_per_class'][i]:>10.3f} "
              f"{metrics['recall_per_class'][i]:>10.3f} "
              f"{metrics['f1_per_class'][i]:>10.3f}")

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = "  " + " " * 18 + "".join(
        [f"{CLASSES[i][:9]:>10}" for i in range(NUM_CLASSES)]
    )
    print(header)
    for i, row in enumerate(metrics['confusion_matrix']):
        row_str = "".join([f"{v:>10}" for v in row])
        print(f"  {CLASSES[i][:18]:<18}{row_str}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    data_dir   = Path(__file__).parent / "data"
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    features_path = data_dir / "features.npz"

    print("=" * 62)
    print("SentinelEdge — Model Training")
    print("=" * 62)

    if not features_path.exists():
        raise FileNotFoundError(
            f"\nFeatures not found: {features_path}"
            f"\nRun 02_feature_engineering.py first."
        )

    # ── Load Data ──────────────────────────────────────────────────────────
    data = np.load(features_path, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int32)
    feature_names = list(data['feature_names'])

    print(f"\nFeature matrix : {X.shape}")
    print(f"Class balance  : {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Train/Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain / Test   : {len(X_train)} / {len(X_test)} samples")

    # ── StandardScaler ─────────────────────────────────────────────────────
    # IMPORTANT: The ESP32 firmware must apply the SAME normalization before
    # passing features to TFLite. Scale params are exported to model_settings.h.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    scaler_params = {
        'mean':          scaler.mean_.tolist(),
        'scale':         scaler.scale_.tolist(),
        'var':           scaler.var_.tolist(),
        'feature_names': feature_names,
        'n_samples_fit': int(scaler.n_samples_seen_),
    }
    with open(models_dir / 'scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"\n✓ Scaler params saved → models/scaler_params.json")

    # ── V1: Isolation Forest ───────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("V1: Isolation Forest (Anomaly Detection — Normal class only)")
    print(f"{'─'*62}")

    X_normal = X_train_scaled[y_train == 0]
    print(f"  Training samples (normal only): {len(X_normal)}")

    iso = IsolationForest(
        n_estimators=100, contamination=0.05,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    t0 = time.perf_counter()
    iso.fit(X_normal)
    print(f"  Training time: {(time.perf_counter()-t0)*1000:.0f}ms")

    iso_pred  = iso.predict(X_test_scaled)
    is_anom   = (iso_pred == -1).astype(int)   # 1 = anomaly
    is_fault  = (y_test   != 0).astype(int)    # 1 = true fault

    iso_acc = float(accuracy_score(is_fault, is_anom))
    iso_fpr = float(np.sum((is_anom == 1) & (y_test == 0)) / np.sum(y_test == 0))
    print(f"  Binary accuracy (normal vs any fault): {iso_acc*100:.2f}%")
    print(f"  False positive rate                  : {iso_fpr*100:.2f}%")

    with open(models_dir / 'isolation_forest.pkl', 'wb') as f:
        pickle.dump({'model': iso, 'scaler': scaler}, f)
    print(f"  ✓ Saved: models/isolation_forest.pkl")

    # ── V2: Dense Neural Network ───────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("V2: Dense Neural Network (4-Class Fault Classification)")
    print(f"{'─'*62}")

    tf.random.set_seed(RANDOM_STATE)
    model = build_nn()
    model.summary()
    print(f"\n  Total trainable params: {model.count_params():,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8,
            min_lr=1e-6, verbose=0,
        ),
    ]

    print(f"\nTraining...")
    t0 = time.perf_counter()
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.perf_counter() - t0

    best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
    print(f"\n  Training time  : {train_time:.2f}s")
    print(f"  Best epoch     : {best_epoch} / {len(history.history['accuracy'])}")
    print(f"  Final val acc  : {max(history.history['val_accuracy'])*100:.2f}%")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print(f"\nTest set evaluation ({len(X_test_scaled)} samples):")
    metrics = evaluate_nn(model, X_test_scaled, y_test)
    print_metrics(metrics)

    # ── Performance targets (assertions) ──────────────────────────────────
    print(f"\n{'─'*62}")
    print("Performance Targets:")
    print(f"{'─'*62}")

    def check(name, value, target, op, unit="%", scale=100.0):
        val_str = f"{value*scale:.2f}{unit}"
        tgt_str = f"{target*scale:.2f}{unit}"
        ops = {'>=': value >= target, '<': value < target, '<=': value <= target}
        passed = ops[op]
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name:<40} {val_str} ({op} {tgt_str})")
        return passed

    p1 = check("Accuracy",          metrics['accuracy'],            0.85, ">=")
    p2 = check("False Positive Rate", metrics['false_positive_rate'], 0.10, "<")

    if not (p1 and p2):
        print("\n  ⚠  Some targets not met. Consider:")
        print("     - Increasing training data (longer duration per class)")
        print("     - Tuning learning rate or architecture")
        print("     - Running real data collection (synthetic data limitation)")
        raise AssertionError("Model does not meet minimum performance targets.")
    else:
        print(f"\n  ✓ All targets met — proceed to TFLite conversion.")

    # ── Save model + metrics ───────────────────────────────────────────────
    keras_path = models_dir / 'fault_classifier.keras'
    model.save(keras_path)
    print(f"\n✓ Keras model saved: {keras_path}")

    keras_metrics = {
        **metrics,
        'best_epoch':   best_epoch,
        'total_epochs': len(history.history['accuracy']),
        'train_time_s': round(train_time, 2),
        'val_accuracy_history': history.history['val_accuracy'],
    }
    with open(models_dir / 'keras_metrics.json', 'w') as f:
        json.dump(keras_metrics, f, indent=2)
    print(f"✓ Metrics saved: models/keras_metrics.json")


if __name__ == "__main__":
    main()
