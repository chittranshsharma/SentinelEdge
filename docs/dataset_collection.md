# Dataset Collection Protocol — SentinelEdge

This document outlines the strict procedure for replacing the synthetic dataset with real-world vibration data for production use.

## Why Real Data Matters

The default `01_generate_synthetic_data.py` script provides mathematically perfect, easily separable clusters (e.g., pure 25Hz sine waves for normal operation). **Do not use the synthetic model in production.** Real physical systems exhibit mechanical resonance, sensor noise, and environmental harmonics that a synthetic model will fail to classify correctly.

## The Setup

1. **Mounting:** Securely attach the ESP32 and MPU6050 to the chassis of the machine (e.g., a cooling fan, motor casing, or pump). Use rigid mounting (screws or strong epoxy); double-sided tape will dampen high-frequency vibrations and ruin the data.
2. **Firmware:** Flash the data collection firmware to the ESP32.
   ```bash
   cd firmware
   pio run -e collection --target upload
   ```
3. **Connection:** Connect the ESP32 to your PC via USB. Ensure the serial monitor is closed in PlatformIO to free the COM port.

## Collection Procedure

You must collect continuous vibration data for each fault class individually. The provided script `ml/serial_logger.py` automates the CSV writing.

### 1. Normal State (Class 0)
Run the machine under normal, healthy operating conditions.
```bash
python ml/serial_logger.py --port COM3 --label 0 --duration 120 --output data/raw_class_0.csv
```
*Note: Replace `COM3` with your actual serial port (e.g., `/dev/ttyUSB0` on Linux/Mac).*

### 2. Imbalance State (Class 1)
Introduce a physical imbalance. For a fan, tape a small coin to one of the blades.
```bash
python ml/serial_logger.py --port COM3 --label 1 --duration 120 --output data/raw_class_1.csv
```

### 3. Obstruction State (Class 2)
Introduce intermittent physical contact. For a fan, lightly place a piece of stiff cardboard so the blades barely strike it as they spin.
```bash
python ml/serial_logger.py --port COM3 --label 2 --duration 120 --output data/raw_class_2.csv
```

### 4. Loose Mount State (Class 3)
Simulate structural instability. Loosen the mounting screws of the machine or place it on an uneven, vibrating surface.
```bash
python ml/serial_logger.py --port COM3 --label 3 --duration 120 --output data/raw_class_3.csv
```

## Data Validation

After collecting all 4 files, visually inspect the CSVs.
- You should have exactly 6 columns: `ax, ay, az, gx, gy, gz`
- Values should be floating point numbers.
- Sample rate must be consistently ~100Hz (120 seconds = ~12,000 rows per file).

## Retraining the Pipeline

Once the real data is collected, modify the ML pipeline to ingest the real CSVs instead of generating synthetic data.

1. Skip `01_generate_synthetic_data.py`.
2. Update `02_feature_engineering.py` to read your 4 `raw_class_X.csv` files and generate the combined feature dataset.
3. Run `03_train_model.py` to train on the real data.
4. Run `04_convert_to_tflite.py` to generate the new C++ headers.
5. Crucially, run `06_feature_drift_validation.py` (via the firmware `drift_check` environment) to ensure the Python feature calculations still perfectly match the C++ calculations for your new dataset distribution.
