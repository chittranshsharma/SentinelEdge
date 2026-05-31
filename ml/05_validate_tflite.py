#!/usr/bin/env python3
"""
SentinelEdge — TFLite Model Validation
=========================================
Validates the int8 quantized model against the Keras float model.
Simulates the full ESP32 inference pipeline in Python.

Validation gates (HARD FAILURES — fix before flashing firmware):
  1. TFLite accuracy >= 85%
  2. Keras → TFLite accuracy drop < 5%          [KEY: quantization quality check]
  3. False positive rate < 10%

Also verifies the confidence threshold filtering and per-class accuracy.
Updates models/model_report.json with final TFLite metrics.

ESP32 simulation pipeline (mirrors firmware exactly):
  raw features (float)
    → subtract kScalerMean   (from scaler_params.json)
    → divide by kScalerScale
    → quantize: q = round(f / kInputScale) + kInputZeroPoint → clamp int8
    → TFLite invoke (int8)
    → argmax of int8 output = predicted class
    → dequantize: (q - kOutputZeroPoint) * kOutputScale = confidence

If quantization accuracy drop >= 5%:
  → Retrain with more data
  → Or use quantization-aware training (QAT) in 03_train_model.py
"""

import numpy as np
import json, time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────

ML_DIR      = Path(__file__).parent
MODELS_DIR  = ML_DIR / "models"
DATA_DIR    = ML_DIR / "data"

TFLITE_PATH = MODELS_DIR / "fault_model.tflite"
KERAS_PATH  = MODELS_DIR / "fault_classifier.keras"
REPORT_PATH = MODELS_DIR / "model_report.json"

CLASSES = {0: "normal", 1: "imbalance", 2: "obstruction", 3: "loose_mount"}

# Hard validation targets
TARGET_TFLITE_ACC  = 0.85
TARGET_ACC_DROP    = 0.05   # max allowed Keras→TFLite drop
TARGET_MAX_FPR     = 0.10
CONF_THRESHOLD     = 0.75

# ── Quantization Helpers ───────────────────────────────────────────────────────

def quantize_features(x_norm: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """
    Quantize normalized float features → int8.
    Mirrors exactly what firmware/src/main.cpp does before calling interpreter->Invoke().

    Formula: q = clamp(round(x / scale) + zp, -128, 127)

    Args:
        x_norm : float32 array, already normalized (subtract mean, divide by scale)
        scale  : kInputScale from model_settings.h
        zp     : kInputZeroPoint from model_settings.h

    Returns:
        int8 array, clipped to [-128, 127]
    """
    q = np.round(x_norm / scale).astype(np.int32) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_output(q: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """
    Dequantize int8 model output → float probabilities.
    Formula: prob = (q - zp) * scale

    Note: For class selection, skip dequantization — argmax of raw int8 is equivalent.
    This is only needed for confidence score computation.
    """
    return (q.astype(np.float32) - zp) * scale


# ── TFLite Inference ───────────────────────────────────────────────────────────

def run_tflite_batch(
    tflite_path: str,
    X_int8: np.ndarray,
) -> tuple:
    """
    Run TFLite int8 inference on a batch.
    Returns (int8 raw outputs, avg_latency_ms_per_sample).
    """
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    n_samples = len(X_int8)
    n_classes = out_det['shape'][-1]
    outputs   = np.empty((n_samples, n_classes), dtype=np.int8)

    t_total = 0.0
    for i in range(n_samples):
        sample = X_int8[i].reshape(inp_det['shape'])  # (1, 42)
        interp.set_tensor(inp_det['index'], sample)
        t0 = time.perf_counter()
        interp.invoke()
        t_total += time.perf_counter() - t0
        outputs[i] = interp.get_tensor(out_det['index'])[0]

    avg_latency_ms = (t_total / n_samples) * 1000.0
    return outputs, avg_latency_ms


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("SentinelEdge — TFLite Validation")
    print("=" * 62)

    # ── Load report (quantization params) ─────────────────────────────────
    with open(REPORT_PATH) as f:
        report = json.load(f)

    in_scale  = report['input_scale']
    in_zp     = report['input_zero_point']
    out_scale = report['output_scale']
    out_zp    = report['output_zero_point']
    keras_acc = report['metrics']['keras_accuracy']

    print(f"\nQuantization parameters (from model_report.json):")
    print(f"  Input  scale / zero_point : {in_scale:.8f} / {in_zp}")
    print(f"  Output scale / zero_point : {out_scale:.8f} / {out_zp}")
    print(f"  Keras accuracy (reference): {keras_acc*100:.2f}%")

    # ── Load features and reconstruct test split ───────────────────────────
    data = np.load(DATA_DIR / 'features.npz', allow_pickle=True)
    X    = data['X'].astype(np.float32)
    y    = data['y'].astype(np.int32)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply StandardScaler normalization (same as training)
    with open(MODELS_DIR / 'scaler_params.json') as f:
        sp = json.load(f)
    mean_v  = np.array(sp['mean'],  dtype=np.float32)
    scale_v = np.array(sp['scale'], dtype=np.float32)
    X_test_norm = (X_test - mean_v) / scale_v

    print(f"\nTest set: {len(X_test_norm)} samples")

    # ── Path A: Keras float inference (reference) ──────────────────────────
    print("\n── Keras (float32) inference ──────────────────────────────────")
    km = keras.models.load_model(KERAS_PATH)
    t0 = time.perf_counter()
    keras_proba = km.predict(X_test_norm, verbose=0)
    keras_time  = (time.perf_counter() - t0) / len(X_test_norm) * 1000
    keras_preds = np.argmax(keras_proba, axis=1)
    keras_acc_v = float(accuracy_score(y_test, keras_preds))
    print(f"  Accuracy     : {keras_acc_v*100:.2f}%")
    print(f"  Latency/smpl : {keras_time:.3f}ms  (Python CPU, not ESP32)")

    # ── Path B: TFLite int8 (full ESP32 simulation) ────────────────────────
    print("\n── TFLite int8 inference (ESP32 pipeline simulation) ──────────")

    print("  Step 1 — Quantize normalized features → int8")
    X_int8 = quantize_features(X_test_norm, in_scale, in_zp)
    pct_saturated = np.mean((X_int8 == -128) | (X_int8 == 127)) * 100
    print(f"    Int8 range       : [{X_int8.min()}, {X_int8.max()}]")
    print(f"    Saturated values : {pct_saturated:.2f}%  (warn if > 5%)")
    if pct_saturated > 5.0:
        print(f"    ⚠  High saturation may indicate calibration issue")

    print("  Step 2 — TFLite int8 inference")
    tflite_raw, tflite_lat = run_tflite_batch(str(TFLITE_PATH), X_int8)
    print(f"    Avg latency/smpl : {tflite_lat:.3f}ms  (Python interpreter)")
    print(f"    NOTE: ESP32 at 240MHz is ~3-8× faster than this Python estimate")

    print("  Step 3 — Argmax on int8 output (no dequantization needed)")
    tflite_preds = np.argmax(tflite_raw, axis=1)
    tflite_acc   = float(accuracy_score(y_test, tflite_preds))

    print("  Step 4 — Dequantize output for confidence scores")
    tflite_proba = dequantize_output(tflite_raw, out_scale, out_zp)
    # Renormalize so confidences sum to 1 (dequant approximation may not)
    tflite_proba_sm = np.exp(tflite_proba) / np.exp(tflite_proba).sum(axis=1, keepdims=True)
    max_conf = tflite_proba_sm.max(axis=1)

    # ── Results ────────────────────────────────────────────────────────────
    acc_drop = keras_acc_v - tflite_acc
    fpr_mask = (y_test == 0)
    tflite_fpr = float(np.sum((tflite_preds != 0) & fpr_mask) / fpr_mask.sum())

    print(f"\n{'='*62}")
    print("ACCURACY COMPARISON")
    print(f"{'='*62}")
    print(f"  Keras (float32)   : {keras_acc_v*100:.2f}%")
    print(f"  TFLite (int8)     : {tflite_acc*100:.2f}%")
    print(f"  Drop              : {acc_drop*100:.2f}pp  (target: < {TARGET_ACC_DROP*100:.0f}pp)")

    print(f"\nPer-class accuracy:")
    print(f"  {'Class':<18} {'Keras':>8} {'TFLite':>8} {'Drop':>8}")
    print(f"  {'─'*44}")
    for label, name in CLASSES.items():
        mask     = y_test == label
        k_acc_c  = float(accuracy_score(y_test[mask], keras_preds[mask]))
        t_acc_c  = float(accuracy_score(y_test[mask], tflite_preds[mask]))
        drop_c   = k_acc_c - t_acc_c
        print(f"  {name:<18} {k_acc_c*100:>7.1f}% {t_acc_c*100:>7.1f}% "
              f"{drop_c*100:>+7.1f}pp")

    # Confidence filtering
    high_conf = max_conf >= CONF_THRESHOLD
    hc_acc = float(accuracy_score(y_test[high_conf], tflite_preds[high_conf])) if high_conf.sum() > 0 else 0.0
    print(f"\nConfidence threshold (>= {CONF_THRESHOLD}):")
    print(f"  Retained : {high_conf.sum():,} / {len(y_test):,} "
          f"samples ({high_conf.mean()*100:.1f}%)")
    print(f"  Accuracy : {hc_acc*100:.2f}%  (should be higher than overall)")

    # Confusion matrix
    cm = confusion_matrix(y_test, tflite_preds)
    print(f"\nTFLite Confusion Matrix (rows=actual, cols=predicted):")
    header = "  " + " " * 18 + "".join([f"{CLASSES[i][:9]:>10}" for i in range(4)])
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i]:<18}" + "".join([f"{v:>10}" for v in row]))

    # ── Validation gates ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("VALIDATION GATES")
    print(f"{'='*62}")

    all_pass = True

    def gate(name, value, target, op, scale=100, unit="%"):
        nonlocal all_pass
        v_str = f"{value*scale:.2f}{unit}"
        t_str = f"{target*scale:.2f}{unit}"
        ops = {'>=': value >= target, '<': value < target}
        ok = ops[op]
        if not ok:
            all_pass = False
        sym = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {sym}  {name:<42} {v_str} ({op} {t_str})")
        return ok

    gate("TFLite Accuracy",             tflite_acc,  TARGET_TFLITE_ACC, ">=")
    gate("Keras→TFLite Accuracy Drop",  acc_drop,    TARGET_ACC_DROP,   "<")
    gate("False Positive Rate",         tflite_fpr,  TARGET_MAX_FPR,    "<")

    if not all_pass:
        print(f"\n  ✗ VALIDATION FAILED")
        print(f"  Remediation options:")
        print(f"    1. Retrain with more data or longer duration per class")
        print(f"    2. Use quantization-aware training (QAT) — modify 03_train_model.py")
        print(f"    3. Increase model capacity (Dense 32→64) — check size stays < 100KB")
        raise SystemExit(1)
    else:
        print(f"\n  ✓ ALL GATES PASSED — model ready for ESP32 deployment")

    # ── Update model_report.json ───────────────────────────────────────────
    report['metrics']['tflite_accuracy']         = round(tflite_acc, 6)
    report['metrics']['accuracy_drop']           = round(acc_drop, 6)
    report['metrics']['tflite_fpr']              = round(tflite_fpr, 6)
    report['metrics']['tflite_latency_ms_python'] = round(tflite_lat, 4)
    report['metrics']['high_conf_retention_pct']  = round(float(high_conf.mean()) * 100, 2)
    report['metrics']['high_conf_accuracy']        = round(hc_acc, 6)

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ model_report.json updated: {REPORT_PATH}")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("FINAL MODEL REPORT")
    print(f"{'='*62}")
    print(f"  Model size (flash)        : {report['model_size_kb']:.2f} KB")
    print(f"  Keras accuracy            : {keras_acc_v*100:.2f}%")
    print(f"  TFLite accuracy (int8)    : {tflite_acc*100:.2f}%")
    print(f"  Accuracy drop             : {acc_drop*100:.2f}pp")
    print(f"  False positive rate       : {tflite_fpr*100:.2f}%")
    print(f"  Confidence >= 75% rate    : {high_conf.mean()*100:.1f}%")
    print(f"  Confidence >= 75% accuracy: {hc_acc*100:.2f}%")
    print(f"  Estimated SRAM usage      : "
          f"{report['estimated_esp32']['estimated_total_sram_kb']:.1f} KB / 520 KB")
    print(f"  Input scale               : {in_scale:.8f}")
    print(f"  Input zero_point          : {in_zp}")
    print(f"\n✓ Phase 1 complete. Proceed to Phase 2: ESP32 Firmware.")


if __name__ == "__main__":
    main()
