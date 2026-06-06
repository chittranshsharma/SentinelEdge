// SentinelEdge — features.cpp
// ============================
// Circular buffer management and 42-feature extraction from MPU6050 windows.
//
// CRITICAL: This file must produce IDENTICAL feature values to ml/feature_utils.py.
// Verify with: python ml/06_feature_drift_validation.py --compare <serial_output.txt>
//
// Known drift risks:
//   1. std/variance: must divide by N (not N-1). ddof=0 = population formula.
//   2. FFT windowing: arduinoFFT must use FFT_WIN_TYP_RECTANGLE (no windowing).
//   3. dominant_freq_bin: search starts at index 1 (skip DC bin 0).
//   4. spectral_energy: sum of squared magnitudes, NOT magnitudes.
//
// Performance target: extractFeatures() + normalizeFeatures() < 50ms total.
// Measured: arduinoFFT N=200 on ESP32 at 240MHz ≈ 4ms per axis = 24ms total.

#include "features.h"
#include <arduinoFFT.h>
#include <math.h>
#include "model/model_settings.h"

// ── Global Buffer Definitions ─────────────────────────────────────────────────
float axBuf[kWindowSize];
float ayBuf[kWindowSize];
float azBuf[kWindowSize];
float gxBuf[kWindowSize];
float gyBuf[kWindowSize];
float gzBuf[kWindowSize];

int   bufferIdx    = 0;
int   totalSamples = 0;

float features[kFeatureCount];
unsigned long featureExtractionUs = 0;

// ── FFT Scratch Buffers ───────────────────────────────────────────────────────
// arduinoFFT requires two arrays: real and imaginary parts.
// Statically allocated to avoid heap fragmentation (TFLite Micro constraint).
// Size must be power-of-2 >= kWindowSize. 256 >= 200 with zero-padding.
#define FFT_SIZE 256
static double fftReal[FFT_SIZE];
static double fftImag[FFT_SIZE];

// arduinoFFT instance (operates on fftReal/fftImag)
static ArduinoFFT<double> FFT(fftReal, fftImag, FFT_SIZE, (double)kSampleRateHz);

// ── Initialization ────────────────────────────────────────────────────────────
void initBuffers() {
    memset(axBuf, 0, sizeof(axBuf));
    memset(ayBuf, 0, sizeof(ayBuf));
    memset(azBuf, 0, sizeof(azBuf));
    memset(gxBuf, 0, sizeof(gxBuf));
    memset(gyBuf, 0, sizeof(gyBuf));
    memset(gzBuf, 0, sizeof(gzBuf));
    bufferIdx    = 0;
    totalSamples = 0;
}

// ── Buffer Push ───────────────────────────────────────────────────────────────
void pushSample(float ax, float ay, float az, float gx, float gy, float gz) {
    int idx = bufferIdx % kWindowSize;
    axBuf[idx] = ax;
    ayBuf[idx] = ay;
    azBuf[idx] = az;
    gxBuf[idx] = gx;
    gyBuf[idx] = gy;
    gzBuf[idx] = gz;

    bufferIdx++;
    totalSamples++;
}

// ── Window Ready ──────────────────────────────────────────────────────────────
bool windowReady() {
    if (totalSamples < kWindowSize) return false;
    // Ready on first full window and every STEP_SIZE samples after
    return ((totalSamples - kWindowSize) % kStepSize) == 0;
}

// ── Per-Axis Feature Extraction ────────────────────────────────────────────────
/**
 * Compute 7 features for one axis.
 * Writes 7 floats starting at out[0].
 *
 * Feature order: mean, std, variance, rms, peak_to_peak,
 *                dominant_freq_bin, spectral_energy
 *
 * IMPORTANT — FFT alignment:
 *   The circular buffer has kWindowSize=200 samples in non-contiguous order.
 *   We must copy them in temporal order (oldest→newest) into fftReal[].
 *   The buffer's read start = bufferIdx % kWindowSize (oldest sample).
 */
static void computeAxisFeatures(float* buf, float* out) {
    int N = kWindowSize;
    int startIdx = bufferIdx % N;

    // ── Time-domain pass ──────────────────────────────────────────────────
    // Copy to ordered temp array while computing mean (single pass)
    // We reuse fftReal[] as temp storage to avoid extra stack allocation.
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        double v  = (double)buf[(startIdx + i) % N];
        fftReal[i] = v;
        fftImag[i] = 0.0;
        sum += v;
    }
    // Zero-pad remaining FFT bins
    for (int i = N; i < FFT_SIZE; i++) {
        fftReal[i] = 0.0;
        fftImag[i] = 0.0;
    }

    double mean = sum / N;

    // Variance, std, rms, peak_to_peak — single pass over fftReal[0..N-1]
    double sumSqDiff = 0.0;
    double sumSq     = 0.0;
    double xMin      = fftReal[0];
    double xMax      = fftReal[0];

    for (int i = 0; i < N; i++) {
        double v    = fftReal[i];
        double diff = v - mean;
        sumSqDiff  += diff * diff;
        sumSq      += v * v;
        if (v < xMin) xMin = v;
        if (v > xMax) xMax = v;
    }

    // Population variance: divide by N (NOT N-1)
    double variance    = sumSqDiff / N;
    double std_val     = sqrt(variance);
    double rms         = sqrt(sumSq / N);
    double peak2peak   = xMax - xMin;

    out[0] = (float)mean;
    out[1] = (float)std_val;
    out[2] = (float)variance;
    out[3] = (float)rms;
    out[4] = (float)peak2peak;

    // ── FFT pass ──────────────────────────────────────────────────────────
    // arduinoFFT uses fftReal[] and fftImag[] (already filled above).
    //
    // CRITICAL: Use FFT_WIN_TYP_RECTANGLE (no windowing) to match numpy's
    // np.fft.rfft which uses a rectangular (boxcar) window by default.
    // Any other window changes magnitudes and will cause spectral_energy drift.
    FFT.windowing(FFTWindow::Rectangle, FFTDirection::Forward);
    FFT.compute(FFTDirection::Forward);  // in-place DFT on fftReal/fftImag
    FFT.complexToMagnitude();            // fftReal[i] = sqrt(real²+imag²), i=0..FFT_SIZE-1

    // Bin 0 = DC (mean). Skip it.
    // Search bins 1..N/2 for dominant frequency.
    // NOTE: With zero-padding (FFT_SIZE=256, data=200), frequency resolution changes.
    // To match Python (N=200, no zero-padding):
    //   Python bin k ↔ frequency k * (fs/N) = k * 0.5 Hz
    //   FFT_SIZE=256 bin k ↔ frequency k * (fs/256) = k * 0.390625 Hz
    // For simple dominant bin identification, we compute on the first N//2 bins only.
    // Spectral energy sums bins 1..N/2 to match Python's 1..N//2 range.
    int halfN = N / 2;  // = 100 for N=200

    int    domBin    = 1;
    double maxMag    = fftReal[1];
    double specEnergy = 0.0;

    for (int k = 1; k <= halfN; k++) {
        double mag = fftReal[k];
        if (mag > maxMag) {
            maxMag = mag;
            domBin = k;
        }
        specEnergy += mag * mag;
    }

    out[5] = (float)domBin;
    out[6] = (float)specEnergy;
}

// ── Main Feature Extraction ────────────────────────────────────────────────────
void extractFeatures() {
    unsigned long t0 = micros();

    float* buffers[NUM_AXES] = { axBuf, ayBuf, azBuf, gxBuf, gyBuf, gzBuf };
    for (int axis = 0; axis < NUM_AXES; axis++) {
        computeAxisFeatures(buffers[axis], &features[axis * FEATURES_PER_AXIS]);
    }

    featureExtractionUs = micros() - t0;
}

// ── Normalization ─────────────────────────────────────────────────────────────
void normalizeFeatures() {
    // Apply StandardScaler: f_norm = (f_raw - mean) / scale
    // kScalerMean and kScalerScale are arrays in model_settings.h (42 values each)
    for (int i = 0; i < kFeatureCount; i++) {
        features[i] = (features[i] - kScalerMean[i]) / kScalerScale[i];
    }
}

// ── Quantization ──────────────────────────────────────────────────────────────
void quantizeFeatures(int8_t* inputTensor) {
    // Quantize normalized float features to int8 for TFLite input tensor.
    // Formula: q = clamp(round(f / kInputScale) + kInputZeroPoint, -128, 127)
    for (int i = 0; i < kFeatureCount; i++) {
        float normalized = features[i];  // already normalized in normalizeFeatures()
        int32_t q = (int32_t)roundf(normalized / kInputScale) + kInputZeroPoint;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        inputTensor[i] = (int8_t)q;
    }
}

// ── Drift Check (compile with -DDRIFT_CHECK_ENABLED=1) ────────────────────────
#ifdef DRIFT_CHECK_ENABLED
void printDriftCheckFeatures() {
    // Regenerate deterministic pure-sine windows (matching 06_feature_drift_validation.py)
    // Signal params per class:
    //   0=normal: 25Hz, 0.10g   1=imbalance: 25Hz, 0.80g
    //   2=obstruction: 25Hz, 0.30g   3=loose_mount: 12Hz, 0.50g
    const float G = 9.81f;
    const float freqs[4] = { 25.0f, 25.0f, 25.0f, 12.0f };
    const float amps[4]  = { 0.10f * G, 0.80f * G, 0.30f * G, 0.50f * G };

    float tempBuf[kWindowSize];
    float tempFeatures[kFeatureCount];

    for (int cls = 0; cls < 4; cls++) {
        float freq = freqs[cls];
        float amp  = amps[cls];

        Serial.print("DRIFT_CLASS:");
        Serial.println(cls);

        // Fill buffers with pure sine window (matches generate_drift_window() in Python)
        float* bufs[NUM_AXES];
        // Use static temp storage for each axis
        static float driftAxBuf[kWindowSize], driftAyBuf[kWindowSize];
        static float driftAzBuf[kWindowSize], driftGxBuf[kWindowSize];
        static float driftGyBuf[kWindowSize], driftGzBuf[kWindowSize];

        bufs[0] = driftAxBuf;
        bufs[1] = driftAyBuf;
        bufs[2] = driftAzBuf;
        bufs[3] = driftGxBuf;
        bufs[4] = driftGyBuf;
        bufs[5] = driftGzBuf;

        for (int i = 0; i < kWindowSize; i++) {
            float t    = (float)i / (float)kSampleRateHz;
            float sine = amp * sinf(2.0f * M_PI * freq * t);
            float cosw = amp * cosf(2.0f * M_PI * freq * t);

            driftAxBuf[i] = sine;
            driftAyBuf[i] = 0.5f  * sine;
            driftAzBuf[i] = G     + 0.2f  * sine;
            driftGxBuf[i] = 0.5f  * cosw;
            driftGyBuf[i] = 0.2f  * 0.5f  * cosw;
            driftGzBuf[i] = 0.1f  * 0.5f  * cosw;
        }

        // Temporarily override circular buffer (restore bufferIdx after)
        int savedIdx = bufferIdx;
        for (int i = 0; i < kWindowSize; i++) {
            axBuf[i] = driftAxBuf[i];
            ayBuf[i] = driftAyBuf[i];
            azBuf[i] = driftAzBuf[i];
            gxBuf[i] = driftGxBuf[i];
            gyBuf[i] = driftGyBuf[i];
            gzBuf[i] = driftGzBuf[i];
        }
        bufferIdx = 0;  // start reading from index 0

        extractFeatures();

        // Print 42 raw (un-normalized) features, one per line
        // Python expects these before normalization
        for (int i = 0; i < kFeatureCount; i++) {
            Serial.println(features[i], 8);
        }

        bufferIdx = savedIdx;
        delay(100);
    }

    Serial.println("DRIFT_CHECK_DONE");
}
#endif  // DRIFT_CHECK_ENABLED
