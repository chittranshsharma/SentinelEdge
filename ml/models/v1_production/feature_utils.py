"""
SentinelEdge — Canonical Feature Extraction
=============================================
THIS IS THE SINGLE SOURCE OF TRUTH for feature extraction.

Both the Python training pipeline (02_feature_engineering.py) and the drift
validation tool (06_feature_drift_validation.py) import from this module.
Any change here must be reflected identically in firmware/src/features.cpp.

Feature set (7 features × 6 axes = 42 features total):
  Index  Feature           Formula
  ─────  ───────────────   ─────────────────────────────────────────────────
  0      mean              sum(x) / N
  1      std               sqrt(sum((x-mean)²) / N)           [population, ddof=0]
  2      variance          sum((x-mean)²) / N                 [population, ddof=0]
  3      rms               sqrt(sum(x²) / N)
  4      peak_to_peak      max(x) - min(x)
  5      dominant_freq_bin argmax(|rfft(x)|[1:N//2+1]) + 1   [1-indexed, skips DC bin 0]
  6      spectral_energy   sum(|rfft(x)|[1:N//2+1]²)         [unnormalized, no windowing]

Axes order: ax, ay, az, gx, gy, gz
Feature order: all 7 features for ax, then ay, az, gx, gy, gz

CRITICAL IMPLEMENTATION NOTES (prevent Python↔ESP32 drift):
  1. Population std/variance (ddof=0): divide by N, NOT N-1.
     C++ firmware uses: `variance = sum_sq / N` (not N-1).
     Mismatch causes std drift of ~0.25% on N=200 windows.

  2. FFT: np.fft.rfft (no windowing = rectangular window).
     Arduino firmware must use FFT_WIN_TYP_RECTANGLE in arduinoFFT.
     Any other window (Hann, Hamming) changes magnitudes by 50%.

  3. FFT normalization: np.fft.rfft is UNNORMALIZED.
     arduinoFFT compute(FFT_MAGNITUDE) is also unnormalized.
     Magnitudes match directly — no scaling factor needed.

  4. dominant_freq_bin: 1-indexed (bin 1 = fs/N Hz).
     Bin 0 is DC (mean). We skip it. Return value is the bin index,
     not the actual frequency in Hz.

  5. spectral_energy: sum of SQUARED magnitudes, not magnitudes.
     This is proportional to Parseval's theorem energy estimate.

  6. float64 in Python, float32 in C++.
     Expect <0.01% relative error from precision difference alone.
"""

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

WINDOW_SIZE = 200
STEP_SIZE   = 100
AXES        = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
NUM_AXES    = len(AXES)     # 6

FEATURE_NAMES_PER_AXIS = [
    'mean',
    'std',
    'variance',
    'rms',
    'peak_to_peak',
    'dominant_freq_bin',
    'spectral_energy',
]
NUM_FEATURES_PER_AXIS = len(FEATURE_NAMES_PER_AXIS)  # 7
FEATURE_COUNT = NUM_AXES * NUM_FEATURES_PER_AXIS       # 42

SAMPLE_RATE_HZ = 100
FREQ_RESOLUTION_HZ = SAMPLE_RATE_HZ / WINDOW_SIZE     # 0.5 Hz per bin

# ── Feature Extraction ─────────────────────────────────────────────────────────

def extract_features_window(window: np.ndarray) -> np.ndarray:
    """
    Extract 42 features from a single (WINDOW_SIZE × 6) window.

    Args:
        window: np.ndarray, shape (WINDOW_SIZE, 6), dtype float.
                Columns must be in order: ax, ay, az, gx, gy, gz.

    Returns:
        np.ndarray, shape (42,), dtype float64.
        Feature order: [ax_mean, ax_std, ..., ax_spectral_energy,
                        ay_mean, ..., gz_spectral_energy]

    Raises:
        AssertionError if window shape is wrong.
    """
    if window.shape != (WINDOW_SIZE, NUM_AXES):
        raise ValueError(
            f"Expected window shape ({WINDOW_SIZE}, {NUM_AXES}), got {window.shape}"
        )

    N = WINDOW_SIZE
    features = np.empty(FEATURE_COUNT, dtype=np.float64)
    idx = 0

    for col in range(NUM_AXES):
        x = window[:, col].astype(np.float64)

        # ── Time-domain ──────────────────────────────────────────────────
        # mean: sum(x) / N
        # Implementation: np.sum for numerical stability; matches C++ loop sum
        mean = np.sum(x) / N

        # Population variance: sum((x - mean)^2) / N  [ddof=0]
        # DO NOT use np.var(x) without ddof=0 — default may vary.
        diff = x - mean
        variance = np.sum(diff * diff) / N

        # Population std: sqrt(variance)
        std = np.sqrt(variance)

        # RMS: sqrt(sum(x^2) / N)
        # Note: RMS ≠ std unless mean=0. Both are computed separately.
        rms = np.sqrt(np.sum(x * x) / N)

        # Peak-to-peak: max - min
        peak_to_peak = x.max() - x.min()

        # ── Frequency-domain (Handled below) ─────────────────────────────
        # ── Store Time Domain ─────────────────────────────────────────────
        features[idx + 0] = mean
        features[idx + 1] = std
        features[idx + 2] = variance
        features[idx + 3] = rms
        features[idx + 4] = peak_to_peak

        # ── FFT pass ────────────────────────────────────────────────────────
        # C++ compatibility:
        # 1. arduinoFFT uses 256 points (zero-padded from 200).
        # 2. To prevent DC leakage during zero-padding, we must subtract the mean first.
        # 3. C++ sums bins 1 to 100.
        mean_val = features[idx+0]
        x_zero_mean = x - mean_val
        
        x_padded = np.zeros(256, dtype=np.float64)
        x_padded[:N] = x_zero_mean
        
        fft_vals = np.fft.rfft(x_padded)
        magnitudes = np.abs(fft_vals)

        # Skip bin 0 (DC). Use bins 1 through 100 inclusive (indices 1:101)
        dom_bin = np.argmax(magnitudes[1:101]) + 1
        features[idx+5] = dom_bin

        spec_energy = np.sum(magnitudes[1:101] ** 2)
        features[idx+6] = spec_energy
        idx += NUM_FEATURES_PER_AXIS

    return features


def build_feature_names() -> list:
    """Return ordered list of all 42 feature names."""
    return [f"{axis}_{feat}" for axis in AXES for feat in FEATURE_NAMES_PER_AXIS]


def bin_to_frequency_hz(bin_index: float) -> float:
    """Convert FFT bin index to frequency in Hz."""
    return bin_index * FREQ_RESOLUTION_HZ


def sliding_window_features(
    data: np.ndarray,
    labels: np.ndarray,
) -> tuple:
    """
    Apply sliding window feature extraction over a full dataset.

    Args:
        data:   np.ndarray shape (N_total, 6), axes: ax ay az gx gy gz
        labels: np.ndarray shape (N_total,), integer class labels

    Returns:
        X: np.ndarray shape (n_windows, 42)
        y: np.ndarray shape (n_windows,), integer labels (from window start)
    """
    n_total   = len(data)
    n_windows = (n_total - WINDOW_SIZE) // STEP_SIZE + 1

    X = np.empty((n_windows, FEATURE_COUNT), dtype=np.float64)
    y = np.empty(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * STEP_SIZE
        end   = start + WINDOW_SIZE
        X[i]  = extract_features_window(data[start:end])
        y[i]  = int(labels[start])

    return X, y
