#!/usr/bin/env python3
"""
SentinelEdge — Feature Drift Validation
==========================================
Generates reference feature values from deterministic (noise-free) test windows.
The ESP32 firmware must produce identical values for these exact same inputs.

PURPOSE:
  Python trains the model. C++ runs inference. If they compute different feature
  values from identical sensor windows, the model receives an out-of-distribution
  input and produces wrong predictions — SILENTLY.

  This script generates:
    data/drift_reference.json — Python's feature values for 4 deterministic windows

  The firmware (when compiled with DRIFT_CHECK=1) reproduces the same windows
  and prints all 42 features via Serial. Compare the two sets of values.

  Maximum allowed relative deviation: 1%
    |python_val - esp32_val| / max(|python_val|, 1e-10) < 0.01

KNOWN DRIFT SOURCES (ordered by impact):
  1. FFT spectral_energy: arduinoFFT magnitude normalization.
     FIX: Use FFT_WIN_TYP_RECTANGLE in arduinoFFT (no windowing).
          Both numpy rfft and arduinoFFT are then unnormalized.

  2. std/variance: sample (ddof=1) vs population (ddof=0).
     FIX: C++ must divide by N (not N-1).
          For N=200: error = (N)/(N-1) = 1.005 = 0.5% (small but compounding).

  3. float32 vs float64 precision.
     EXPECTED: ~0.001% relative error. Acceptable, no fix needed.

  4. dominant_freq_bin: off-by-one in FFT bin indexing.
     FIX: Skip bin 0 (DC). First non-DC bin is bin 1.
          In C++: start search from index 1, not 0.

HOW TO RUN ESP32 DRIFT CHECK:
  1. In platformio.ini, add: build_flags = -D DRIFT_CHECK_ENABLED=1
  2. Flash firmware
  3. Open serial monitor (115200 baud)
  4. Send character 'd' → firmware prints 42 features for each class window
  5. Run: python 06_feature_drift_validation.py --compare <serial_output.txt>
     (or manually compare with drift_reference.json)
"""

import numpy as np
import json
from pathlib import Path
import argparse
import sys

from feature_utils import (
    WINDOW_SIZE, AXES, FEATURE_COUNT, NUM_AXES,
    FEATURE_NAMES_PER_AXIS, NUM_FEATURES_PER_AXIS,
    extract_features_window,
    bin_to_frequency_hz,
)

G = 9.81
CLASSES = {0: "normal", 1: "imbalance", 2: "obstruction", 3: "loose_mount"}
MAX_RELATIVE_ERROR = 0.01  # 1% tolerance


# ── Deterministic Test Window Generator ───────────────────────────────────────

def generate_drift_window(label: int) -> np.ndarray:
    """
    Generate a deterministic, noise-free test window for drift comparison.

    Signal: pure sine waves, NO random components.
    This makes the window EXACTLY reproducible in C++ with the same formula.

    C++ equivalent (see firmware/src/features.cpp, driftCheckWindow()):
      float t = (float)i / 100.0f;  // i in [0, 199]
      float sine  = amp * sinf(2.0f * M_PI * freq * t);
      float cosw  = amp * cosf(2.0f * M_PI * freq * t);
      ax[i] = sine;
      ay[i] = 0.5f * sine;
      az[i] = 9.81f + 0.2f * sine;
      gx[i] = 0.5f * cosw;
      gy[i] = 0.2f * 0.5f * cosw;
      gz[i] = 0.1f * 0.5f * cosw;

    Dominant FFT bins (freq * N / fs):
      Normal/Imbalance/Obstruction: 25Hz → bin 50 (25 * 200 / 100 = 50)
      Loose Mount                 : 12Hz → bin 24 (12 * 200 / 100 = 24)
    """
    params = {
        0: (25.0, 0.10 * G),  # Normal
        1: (25.0, 0.80 * G),  # Imbalance
        2: (25.0, 0.30 * G),  # Obstruction
        3: (12.0, 0.50 * G),  # Loose Mount
    }
    freq, amp = params[label]

    t    = np.arange(WINDOW_SIZE, dtype=np.float64) / 100.0
    sine = amp * np.sin(2.0 * np.pi * freq * t)
    cosw = amp * np.cos(2.0 * np.pi * freq * t)

    ax = sine
    ay = 0.5  * sine
    az = G    + 0.2  * sine
    gx = 0.5  * cosw
    gy = 0.2  * 0.5  * cosw
    gz = 0.1  * 0.5  * cosw

    return np.column_stack([ax, ay, az, gx, gy, gz])


def features_to_dict(features: np.ndarray) -> dict:
    """Convert flat feature array (42,) to named dict for JSON output."""
    result = {}
    idx = 0
    for axis in AXES:
        for feat in FEATURE_NAMES_PER_AXIS:
            result[f"{axis}_{feat}"] = float(features[idx])
            idx += 1
    return result


# ── Comparison Mode ────────────────────────────────────────────────────────────

def compare_with_esp32(esp32_file: str, reference: dict):
    """
    Compare ESP32 serial output with Python reference values.

    Expected serial format (one feature per line):
      DRIFT_CLASS:0
      0.981000
      0.000000
      ...  (42 lines of float values, then next class)

    Prints pass/fail for each feature with relative error.
    """
    with open(esp32_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\nComparing ESP32 output vs Python reference...")
    print(f"  File: {esp32_file}")
    print(f"  Tolerance: {MAX_RELATIVE_ERROR*100:.0f}% relative error")
    print()

    current_class = None
    esp32_vals = []
    all_pass = True

    for line in lines:
        if line.startswith("DRIFT_CLASS:"):
            # Process previous class if any
            if current_class is not None and esp32_vals:
                all_pass &= compare_class(current_class, esp32_vals, reference)
            current_class = int(line.split(":")[1])
            esp32_vals = []
        else:
            try:
                esp32_vals.append(float(line))
            except ValueError:
                pass  # skip non-numeric lines

    # Process last class
    if current_class is not None and esp32_vals:
        all_pass &= compare_class(current_class, esp32_vals, reference)

    if all_pass:
        print(f"\n✓ ALL DRIFT CHECKS PASSED — Python and ESP32 features agree within {MAX_RELATIVE_ERROR*100:.0f}%")
    else:
        print(f"\n✗ DRIFT DETECTED — Fix C++ feature extraction before deploying model")
        print(f"  See comments in ml/06_feature_drift_validation.py for known drift sources")


def compare_class(label: int, esp32_vals: list, reference: dict):
    name = CLASSES[label]
    ref_class = reference['windows'][name]
    py_vals = ref_class['features_flat']

    if len(esp32_vals) != FEATURE_COUNT:
        print(f"\n  [{label}] {name}: Expected {FEATURE_COUNT} values, got {len(esp32_vals)}")
        return False

    print(f"  [{label}] {name}:")
    feature_names = [f"{ax}_{fn}" for ax in AXES for fn in FEATURE_NAMES_PER_AXIS]
    max_err = 0.0
    any_fail = False

    for i, (py, esp, fn) in enumerate(zip(py_vals, esp32_vals, feature_names)):
        denom = max(abs(py), 1e-10)
        rel_err = abs(py - esp) / denom
        max_err = max(max_err, rel_err)
        ok = rel_err < MAX_RELATIVE_ERROR
        if not ok:
            any_fail = True
            print(f"    ✗ {fn:<35}: py={py:>12.6f}  esp32={esp:>12.6f}  "
                  f"err={rel_err*100:>6.3f}%")

    if not any_fail:
        print(f"    ✓ All {FEATURE_COUNT} features within {MAX_RELATIVE_ERROR*100:.0f}%  "
              f"(max err: {max_err*100:.4f}%)")

    return not any_fail


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SentinelEdge feature drift validation")
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to ESP32 serial output file for comparison')
    args = parser.parse_args()

    data_dir    = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "drift_reference.json"

    print("=" * 62)
    print("SentinelEdge — Feature Drift Validation")
    print("=" * 62)
    print()
    print("Generating deterministic reference windows (pure sine, no noise)")
    print("These are exactly reproducible in C++ with sinf() / cosf()")
    print()

    reference = {
        "description": (
            "Reference feature values for ESP32 drift comparison. "
            "Windows are deterministic pure sine waves (no random noise). "
            "ESP32 firmware reproduces identical windows via driftCheckWindow()."
        ),
        "window_size":           WINDOW_SIZE,
        "axes":                  AXES,
        "feature_names_per_axis": FEATURE_NAMES_PER_AXIS,
        "feature_count":         FEATURE_COUNT,
        "tolerance_relative":    MAX_RELATIVE_ERROR,
        "drift_notes": {
            "std_variance": (
                "Uses population formula (ddof=0): divide by N. "
                "C++ must NOT use (N-1). Error ~0.5% for N=200."
            ),
            "fft_windowing": (
                "np.fft.rfft has no windowing (rectangular). "
                "arduinoFFT MUST use FFT_WIN_TYP_RECTANGLE. "
                "Hann window changes magnitudes ~50%."
            ),
            "fft_normalization": (
                "np.fft.rfft is unnormalized. "
                "arduinoFFT compute(FFT_MAGNITUDE) is also unnormalized. "
                "Magnitudes should match directly without scaling."
            ),
            "dominant_freq_bin": (
                "1-indexed bin number of max non-DC magnitude. "
                "DC is bin 0 (skip it). Bin 1 = 0.5 Hz, Bin 50 = 25 Hz."
            ),
            "float_precision": (
                "Python uses float64, C++ uses float32. "
                "Expect ~0.001% relative error from precision alone — acceptable."
            ),
        },
        "windows": {}
    }

    all_feature_names = [f"{ax}_{fn}" for ax in AXES for fn in FEATURE_NAMES_PER_AXIS]

    for label, class_name in CLASSES.items():
        freq = {0: 25.0, 1: 25.0, 2: 25.0, 3: 12.0}[label]
        amp  = {0: 0.10, 1: 0.80, 2: 0.30, 3: 0.50}[label]

        window   = generate_drift_window(label)
        features = extract_features_window(window)
        feat_dict = features_to_dict(features)

        # Expected dominant bin for main axis
        expected_bin = int(freq * WINDOW_SIZE / 100)
        actual_bin   = int(features[5])  # ax_dominant_freq_bin

        print(f"  [{label}] {class_name}:")
        print(f"    Signal      : {amp:.2f}g  at {freq:.0f}Hz  "
              f"(expected FFT bin = {expected_bin})")
        print(f"    ax_dom_bin  : {actual_bin}  "
              f"({'✓' if actual_bin == expected_bin else '✗ MISMATCH'})")
        print(f"    ax_mean     : {features[0]:>14.8f}")
        print(f"    ax_std      : {features[1]:>14.8f}")
        print(f"    ax_rms      : {features[3]:>14.8f}")
        print(f"    ax_energy   : {features[6]:>14.2f}")
        print()

        reference["windows"][class_name] = {
            "label":            label,
            "signal_freq_hz":   freq,
            "signal_amp_g":     amp,
            "expected_fft_bin": expected_bin,
            "raw_window_first_8_rows": window[:8].tolist(),
            "features_dict":    feat_dict,
            "features_flat":    features.tolist(),
            "feature_names":    all_feature_names,
        }

    with open(output_path, 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"✓ Drift reference saved: {output_path}")
    print()

    if args.compare:
        if not Path(args.compare).exists():
            print(f"✗ Comparison file not found: {args.compare}")
            sys.exit(1)
        compare_with_esp32(args.compare, reference)
    else:
        print("HOW TO USE FOR ESP32 VALIDATION:")
        print("─" * 62)
        print("1. Add `build_flags = -D DRIFT_CHECK_ENABLED=1` to platformio.ini")
        print("2. Flash firmware, open serial monitor (115200 baud)")
        print("3. Send 'd' → ESP32 prints 4 × 42 features")
        print("4. Save output to a file, e.g., esp32_drift.txt")
        print("5. Run: python 06_feature_drift_validation.py --compare esp32_drift.txt")
        print()
        print("EXPECTED SERIAL FORMAT FROM ESP32:")
        print("─" * 62)
        print("  DRIFT_CLASS:0")
        print("  0.98100001   ← ax_mean")
        print("  0.69370100   ← ax_std")
        print("  ...  (42 values total)")
        print("  DRIFT_CLASS:1")
        print("  ...")
        print()
        print("TOLERANCE: 1% relative error on each feature")
        print("Spectral energy may show larger deviation if arduinoFFT windowing differs.")
        print("See drift_notes in drift_reference.json for root causes.")


if __name__ == "__main__":
    main()
