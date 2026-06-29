# SentinelEdge ML Training Report (Real Data Pipeline)

This report details the final TinyML model trained purely on real hardware data collected from the MPU6050 sensor on the ESP32.

> [!TIP]
> The entire synthetic data pipeline has been replaced. The system is now driven strictly by physical sensor inputs.

## Dataset Summary

The dataset was constructed from 4 CSV files collected directly from the hardware. Boot logs and malformed lines were automatically filtered.

*   **Total valid sample rows:** 41,933
*   **Window configuration:** 200 samples per window, 100 sample step size
*   **Feature vector:** 42 canonical features (time-domain and frequency-domain) per window.

**Windows per class:**
*   `stationary`: 133
*   `movement`: 98
*   `rotation`: 97
*   `shake`: 85
*   **Total Windows:** 413

## Model Architecture

The classifier is a fully connected Dense Neural Network designed for optimal TFLite Micro Int8 quantization:

```text
Input (42 features) 
  -> Dense(32) + ReLU 
  -> Dense(16) + ReLU 
  -> Dense(4) + Softmax
```

*   **Total Parameters:** 1,972
*   **Quantization:** Full Int8 (weights, activations, inputs, and outputs)
*   **Input Scaling:** Handled on-device via `kScalerMean` and `kScalerScale` embedded in firmware.

## Performance Metrics (Test Set)

No artificial thresholds or data leaks were applied. The results below reflect the true generalization of the model on the unseen test split.

*   **Keras (Float32) Accuracy:** 98.80%
*   **TFLite (Int8) Accuracy:** 98.80%
*   **Quantization Drop:** 0.00pp (Perfect parity)
*   **False Positive Rate:** 0.00%

### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **stationary** | 1.00 | 1.00 | 1.00 |
| **movement** | 0.95 | 1.00 | 0.98 |
| **rotation** | 1.00 | 0.95 | 0.97 |
| **shake** | 1.00 | 1.00 | 1.00 |

### Confusion Matrix

![TFLite Int8 Confusion Matrix](C:\Users\chitt\.gemini\antigravity-ide\brain\5a85fe95-ac94-453a-ac20-a603ee440d20\confusion_matrix.png)

## ESP32 Memory Estimates

The quantized model comfortably fits within the strict constraints of the ESP32:

*   **Model Size (Flash):** 6.01 KB (Target was < 100 KB)
*   **Tensor Arena Requirement:** 12.0 KB
*   **Axis Buffers (6x200 floats):** 4.8 KB
*   **Total SRAM Usage Estimate:** ~18.9 KB (3.6% of available 520 KB)

> [!SUCCESS]
> The model has been successfully compiled and flashed to the ESP32 firmware. The system is ready for Phase 6.5: Real Hardware Validation.
