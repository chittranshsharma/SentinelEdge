#!/usr/bin/env python3
import numpy as np

def test_leakage():
    # Simulate a stationary gravity signal (ax = 9.81) with tiny noise
    N = 200
    FFT_SIZE = 256
    x = np.ones(N) * 9.81 + np.random.normal(0, 0.01, N)
    
    # 1. Python exact (no padding)
    fft_exact = np.abs(np.fft.rfft(x))
    energy_exact = np.sum(fft_exact[1:]**2)
    
    # 2. C++ current (zero padded, no mean subtraction)
    x_pad1 = np.zeros(FFT_SIZE)
    x_pad1[:N] = x
    fft_padded1 = np.abs(np.fft.rfft(x_pad1))
    energy_padded1 = np.sum(fft_padded1[1:101]**2)
    
    # 3. C++ fixed (mean subtracted, then zero padded)
    mean = np.mean(x)
    x_pad2 = np.zeros(FFT_SIZE)
    x_pad2[:N] = x - mean
    fft_padded2 = np.abs(np.fft.rfft(x_pad2))
    energy_padded2 = np.sum(fft_padded2[1:101]**2)
    
    print(f"Energy Exact (Python)    : {energy_exact:.4f}")
    print(f"Energy Padded (C++ Bug)  : {energy_padded1:.4f}")
    print(f"Energy Padded (C++ Fixed): {energy_padded2:.4f}")

if __name__ == "__main__":
    test_leakage()
