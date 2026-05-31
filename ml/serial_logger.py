#!/usr/bin/env python3
"""
SentinelEdge — Real Data Serial Logger
=========================================
Captures raw MPU6050 data from ESP32 serial port to CSV for real data collection.

Usage:
  python serial_logger.py --port COM3 --label 0 --duration 300 --out data/real/normal.csv
  python serial_logger.py --port COM3 --label 1 --duration 600 --out data/real/imbalance.csv

Labels:
  0 = normal       (fan running normally, ~20 minutes minimum)
  1 = imbalance    (coin taped to fan blade, ~10 minutes minimum)
  2 = obstruction  (resistance against fan, ~10 minutes minimum)
  3 = loose_mount  (fan on unstable surface, ~10 minutes minimum)

ESP32 must be flashed in DATA_COLLECTION mode (platformio.ini env: collection).
In that mode, main.cpp prints raw sensor data at 100Hz:
  ax,ay,az,gx,gy,gz
  -0.12,0.34,9.81,0.02,-0.01,0.00
  ...

Collection targets:
  Normal:   1200s  (20 min) × 100Hz = 120,000 rows
  Faults:   600s   (10 min) × 100Hz =  60,000 rows each
  Total:    ~300,000 rows

IMPORTANT: After collection, combine files and retrain:
  python serial_logger.py --combine data/real/ --out data/real_vibration.csv
  python 02_feature_engineering.py  (with REAL_DATA=True)
  python 03_train_model.py
  python 04_convert_to_tflite.py
  python 05_validate_tflite.py
"""

import argparse
import csv
import time
import signal
import sys
from pathlib import Path
from datetime import datetime

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("WARNING: pyserial not installed. Run: pip install pyserial")


BAUD_RATE = 115200
EXPECTED_COLS = 6  # ax, ay, az, gx, gy, gz
COLUMNS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label']
LABELS = {0: "normal", 1: "imbalance", 2: "obstruction", 3: "loose_mount"}


class SerialLogger:
    def __init__(self, port: str, label: int, output_path: str, duration: int):
        self.port        = port
        self.label       = label
        self.output_path = Path(output_path)
        self.duration    = duration
        self.running     = True
        self.row_count   = 0
        self.error_count = 0

    def run(self):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial required. Install: pip install pyserial")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSentinelEdge Serial Logger")
        print(f"{'─'*50}")
        print(f"  Port         : {self.port}")
        print(f"  Baud rate    : {BAUD_RATE}")
        print(f"  Label        : {self.label} ({LABELS.get(self.label, 'unknown')})")
        print(f"  Duration     : {self.duration}s  ({self.duration//60}m {self.duration%60}s)")
        print(f"  Target rows  : ~{self.duration * 100:,}")
        print(f"  Output       : {self.output_path}")
        print(f"\n  Press Ctrl+C to stop early.")
        print()

        signal.signal(signal.SIGINT, lambda s, f: self._stop())

        with serial.Serial(self.port, BAUD_RATE, timeout=1.0) as ser, \
             open(self.output_path, 'w', newline='') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
            writer.writeheader()

            start_time = time.time()
            last_status = start_time

            print(f"  Recording... (started {datetime.now().strftime('%H:%M:%S')})")

            while self.running:
                elapsed = time.time() - start_time
                if elapsed >= self.duration:
                    break

                try:
                    line = ser.readline().decode('ascii', errors='ignore').strip()
                    if not line or line.startswith('#'):
                        continue  # skip comments and empty lines

                    parts = line.split(',')
                    if len(parts) != EXPECTED_COLS:
                        self.error_count += 1
                        continue

                    row = {
                        'ax': float(parts[0]),
                        'ay': float(parts[1]),
                        'az': float(parts[2]),
                        'gx': float(parts[3]),
                        'gy': float(parts[4]),
                        'gz': float(parts[5]),
                        'label': self.label,
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    self.row_count += 1

                    # Status update every 5 seconds
                    now = time.time()
                    if now - last_status >= 5.0:
                        pct = elapsed / self.duration * 100
                        rate = self.row_count / elapsed if elapsed > 0 else 0
                        remaining = self.duration - elapsed
                        print(f"  {pct:5.1f}%  |  "
                              f"{self.row_count:>8,} rows  |  "
                              f"{rate:.0f} Hz  |  "
                              f"{remaining:.0f}s remaining  |  "
                              f"{self.error_count} errors")
                        last_status = now

                except ValueError:
                    self.error_count += 1
                except serial.SerialException as e:
                    print(f"\n  Serial error: {e}")
                    break

        total_time = time.time() - start_time
        actual_rate = self.row_count / total_time if total_time > 0 else 0

        print(f"\n{'─'*50}")
        print(f"  Completed!")
        print(f"  Rows captured  : {self.row_count:,}")
        print(f"  Duration       : {total_time:.1f}s")
        print(f"  Actual rate    : {actual_rate:.1f} Hz  (target: 100 Hz)")
        print(f"  Parse errors   : {self.error_count}")
        print(f"  Output         : {self.output_path}")

        if actual_rate < 90:
            print(f"\n  ⚠  Actual rate {actual_rate:.0f}Hz < 100Hz.")
            print(f"     Check: USB buffer, PC load, or ESP32 timing.")

    def _stop(self):
        self.running = False
        print(f"\n  Stopping...")


def combine_files(input_dir: str, output_path: str):
    """Combine multiple per-class CSV files into a single dataset."""
    import pandas as pd

    input_dir_path = Path(input_dir)
    csv_files = sorted(input_dir_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Combining {len(csv_files)} files:")
    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df):,} rows, label={df['label'].iloc[0]}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\n✓ Combined dataset: {output_path}")
    print(f"  Total rows : {len(combined):,}")
    print(f"  Balance    : {combined['label'].value_counts().sort_index().to_dict()}")


def main():
    parser = argparse.ArgumentParser(
        description="SentinelEdge serial data logger"
    )
    subparsers = parser.add_subparsers(dest='command')

    # Record subcommand
    rec = subparsers.add_parser('record', help='Record serial data')
    rec.add_argument('--port',     required=True, help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    rec.add_argument('--label',    required=True, type=int, choices=[0,1,2,3])
    rec.add_argument('--duration', required=True, type=int, help='Duration in seconds')
    rec.add_argument('--out',      required=True, help='Output CSV path')

    # Combine subcommand
    comb = subparsers.add_parser('combine', help='Combine per-class CSV files')
    comb.add_argument('--input', required=True, help='Directory containing per-class CSVs')
    comb.add_argument('--out',   required=True, help='Output combined CSV path')

    args = parser.parse_args()

    if args.command == 'record':
        logger = SerialLogger(args.port, args.label, args.out, args.duration)
        logger.run()
    elif args.command == 'combine':
        combine_files(args.input, args.out)
    else:
        parser.print_help()
        print(f"\nExamples:")
        print(f"  python serial_logger.py record --port COM3 --label 0 --duration 1200 --out data/real/normal.csv")
        print(f"  python serial_logger.py record --port COM3 --label 1 --duration 600  --out data/real/imbalance.csv")
        print(f"  python serial_logger.py combine --input data/real/ --out data/real_vibration.csv")


if __name__ == "__main__":
    main()
