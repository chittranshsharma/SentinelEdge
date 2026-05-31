#!/usr/bin/env python3
"""
SentinelEdge — Synthetic Vibration Data Generator
==================================================
Generates synthetic MPU6050-like vibration data for 4 fault classes.

Physical model (fan motor on a surface):
  Sampling rate  : 100 Hz
  Duration/class : 60 seconds  →  6,000 samples/class
  Total          : 24,000 samples, 4 classes

Fault classes and physical descriptions:
  0 = Normal       Low amplitude vibration (~0.1g), smooth sine at 25Hz.
                   Fan running cleanly, no mechanical faults.

  1 = Imbalance    High amplitude vibration (~0.8g), same 25Hz frequency.
                   Mass imbalance (e.g., tape/debris on blade) causes
                   elevated centrifugal force at rotational frequency.

  2 = Obstruction  Medium amplitude (~0.3g), HIGH noise floor.
                   Physical contact/resistance against rotating assembly
                   produces random impact transients on top of base sine.

  3 = Loose Mount  Medium amplitude (~0.5g), LOWER dominant frequency (12Hz).
                   Fan on unstable surface → chassis resonance at sub-RPM
                   frequencies + harmonic at 24Hz.

Sensor model (MPU6050 at default ±2g, ±250°/s range):
  ax, ay : radial/axial vibration in m/s²
  az     : includes Earth gravity (~9.81 m/s²) + vertical vibration
  gx, gy, gz : angular velocity in deg/s (driven by vibration coupling)

Output: ml/data/synthetic_vibration.csv
Columns: ax, ay, az, gx, gy, gz, label
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Physical Constants ─────────────────────────────────────────────────────────

G              = 9.81       # m/s² per g
FS             = 100        # Sampling rate (Hz)
DURATION       = 60         # Seconds per class
N              = FS * DURATION  # 6000 samples per class

# Dominant vibration frequencies per class
FREQ = {0: 25.0, 1: 25.0, 2: 25.0, 3: 12.0}

# Vibration amplitudes (g → m/s²)
AMP = {0: 0.10 * G, 1: 0.80 * G, 2: 0.30 * G, 3: 0.50 * G}

# Noise standard deviations (m/s²)
# Higher noise = messier signal = obstruction class
NOISE = {0: 0.02 * G, 1: 0.03 * G, 2: 0.25 * G, 3: 0.10 * G}

CLASSES = {0: "normal", 1: "imbalance", 2: "obstruction", 3: "loose_mount"}

# ── Signal Generator ───────────────────────────────────────────────────────────

def generate_class(label: int, seed: int) -> pd.DataFrame:
    """
    Generate N samples of synthetic vibration data for a single fault class.

    Signal model per axis:
      ax = A * sin(2π*f*t + φ) + noise               [primary radial axis]
      ay = 0.5*A * sin(2π*f*t + φ) + noise           [secondary axial axis]
      az = g + 0.2*A * sin(2π*f*t + φ) + noise       [vertical + gravity]
      gx = 0.5*A * cos(2π*f*t + φ) + noise           [angular, 90° phase shift]
      gy = 0.2*A * cos(2π*f*t + φ) + noise           [angular, attenuated]
      gz = 0.1*A * cos(2π*f*t + φ) + noise           [angular, heavily attenuated]

    Loose mount: adds 2nd harmonic to simulate chassis resonance coupling.
    Obstruction: adds random impulses simulating mechanical contact events.
    """
    rng   = np.random.RandomState(seed)
    t     = np.arange(N, dtype=np.float64) / FS
    phase = rng.uniform(0, 2 * np.pi)

    f     = FREQ[label]
    amp   = AMP[label]
    noise = NOISE[label]

    sine = amp * np.sin(2 * np.pi * f * t + phase)
    cosn = amp * np.cos(2 * np.pi * f * t + phase)  # 90° phase-shifted gyro

    # ── Accelerometer ─────────────────────────────────────────────────────
    ax = sine                                    + rng.normal(0, noise,       N)
    ay = 0.5  * sine                             + rng.normal(0, noise * 0.8, N)
    az = G    + 0.2 * sine                       + rng.normal(0, noise * 0.5, N)

    # ── Gyroscope (angular velocity, deg/s) ───────────────────────────────
    # Gyro signal is proportional to vibration amplitude but phase-shifted
    gyro_amp   = 0.5
    gyro_noise = noise * 0.5  # gyro inherently less noisy than accel

    gx =  gyro_amp        * cosn + rng.normal(0, gyro_noise,       N)
    gy = (gyro_amp * 0.4) * cosn + rng.normal(0, gyro_noise * 0.7, N)
    gz = (gyro_amp * 0.2) * cosn + rng.normal(0, gyro_noise * 0.3, N)

    # ── Class-specific physics ────────────────────────────────────────────

    if label == 3:  # Loose Mount — add 2nd harmonic (resonance)
        harmonic_amp  = amp * 0.35
        harmonic_sine = harmonic_amp * np.sin(2 * np.pi * (f * 2) * t + phase)
        ax += harmonic_sine
        ay += harmonic_sine * 0.5
        gx += harmonic_amp * 0.5 * np.cos(2 * np.pi * (f * 2) * t + phase)

    if label == 2:  # Obstruction — add random impact impulses
        # ~10 impacts per 60 seconds = one impact every 6 seconds on average
        n_impulses = rng.randint(8, 14)
        impulse_idx = rng.choice(N, size=n_impulses, replace=False)
        for idx in impulse_idx:
            direction  = rng.choice([-1, 1])
            magnitude  = rng.uniform(0.4, 1.8) * amp
            ax[idx]   += direction * magnitude
            ay[idx]   += direction * magnitude * rng.uniform(0.2, 0.5)
            # Spread impulse slightly (vibration ring-down over ~3 samples)
            for spread in range(1, min(4, N - idx)):
                decay = 0.4 ** spread
                ax[idx + spread] += direction * magnitude * decay
                ay[idx + spread] += direction * magnitude * decay * 0.5

    return pd.DataFrame({
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'label': label,
    })


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    output_dir  = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "synthetic_vibration.csv"

    print("=" * 62)
    print("SentinelEdge — Synthetic Vibration Data Generator")
    print("=" * 62)
    print(f"\nSampling rate : {FS} Hz")
    print(f"Duration/class: {DURATION}s  →  {N:,} samples")
    print(f"Total samples : {N * len(CLASSES):,}")
    print()

    frames = []
    for label, name in CLASSES.items():
        df = generate_class(label, seed=label * 42 + 7)
        frames.append(df)
        dominant_bin = int(FREQ[label] * N / FS)  # expected FFT bin
        print(f"  [{label}] {name:<15}: amp={AMP[label]/G:.1f}g  "
              f"noise={NOISE[label]/G:.2f}g  "
              f"freq={FREQ[label]:.0f}Hz (FFT bin {dominant_bin})")

    dataset = pd.concat(frames, ignore_index=True)
    dataset.to_csv(output_path, index=False, float_format='%.8f')

    print(f"\n{'─'*62}")
    print(f"✓ Saved: {output_path}")
    print(f"  Rows    : {len(dataset):,}")
    print(f"  Columns : {list(dataset.columns)}")
    print(f"  Balance : {dataset['label'].value_counts().sort_index().to_dict()}")

    print(f"\nPer-class signal statistics (ax axis):")
    print(f"  {'Class':<16} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'RMS':>9}")
    print(f"  {'─'*53}")
    for label, name in CLASSES.items():
        s = dataset[dataset['label'] == label]['ax']
        rms = float(np.sqrt((s**2).mean()))
        print(f"  {name:<16} {s.mean():>+9.3f} {s.std():>9.3f} "
              f"{s.min():>9.3f} {s.max():>9.3f} {rms:>9.3f}")


if __name__ == "__main__":
    main()
