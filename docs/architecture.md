# Architecture Details — SentinelEdge

This document dives deep into the technical design decisions and architecture of the SentinelEdge platform.

## 1. Edge Node (ESP32)

### Hardware Constraints
- **Processor:** Dual-core Xtensa 32-bit LX6 @ 240 MHz
- **Memory:** 520 KB SRAM
- **Storage:** 4 MB Flash

### The TFLite Micro Pipeline
The ESP32 runs the entire inference pipeline. The pipeline involves:
1. **Data Acquisition:** Reading raw MPU6050 accelerometer and gyroscope data at exactly 100 Hz.
2. **Buffering:** Maintaining a rolling window of 200 samples (2 seconds) using a circular buffer to minimize memory shifting overhead.
3. **Feature Extraction:** Calculating 42 distinct statistical and spectral features (7 metrics per axis × 6 axes).
4. **Normalization:** Applying StandardScaler transform using parameters extracted directly from the Python training phase, baked into the C++ headers.
5. **Inference:** Invoking the int8 quantized TFLite model to classify the window into one of 4 states (Normal, Imbalance, Obstruction, Loose Mount).

**Memory Optimization Strategy:**
The TFLite Micro tensor arena is statically allocated (`kTensorArenaSize = 12 * 1024` bytes). This avoids heap fragmentation and ensures deterministic memory usage. The total SRAM usage for the ML pipeline is kept well under 100 KB, leaving ample room for the WiFi and MQTT stacks, which are notoriously memory-hungry.

## 2. Gateway Layer (Raspberry Pi)

### Why a Gateway?
In an industrial environment, edge nodes often cannot connect directly to the public internet due to security policies or lack of reliable WiFi coverage. A local gateway acts as an aggregator and security boundary.

### Responsibilities
- **MQTT Broker:** Mosquitto runs locally to receive high-frequency telemetry and anomaly events from all edge nodes.
- **Data Forwarding:** A FastAPI backend subscribes to the local MQTT broker and forwards structured payloads over HTTPS to the Supabase cloud.
- **LLM Enrichment:** When an anomaly is detected, the backend queries the Groq API (using the Llama 3 70B model) to generate a human-readable diagnostic explanation based on the specific sensor readings (e.g., combining high vibration with high temperature).
- **Alerting:** Formats and sends immediate Telegram notifications for high-confidence faults.

## 3. Cloud Infrastructure (Supabase)

Supabase serves as the persistent data store and realtime synchronization engine.
- **PostgreSQL:** Stores `sensor_readings` (heartbeats) and `anomalies`.
- **Realtime (Postgres Changes):** The Next.js dashboard subscribes to `INSERT` events on these tables. When the gateway pushes a new row, Supabase broadcasts it to all connected dashboard clients via WebSockets, ensuring sub-second UI updates without polling.
- **Row Level Security (RLS):** Read-only policies protect data integrity for dashboard clients, while the gateway uses a secure Service Role key for insertions.

## 4. Visualization (Next.js)

The dashboard is built with Next.js App Router for optimal performance.
- **Client Components:** Intensive interactive elements (Leaflet maps, Recharts) are rendered strictly on the client side (`ssr: false` for Leaflet to prevent `window is not defined` errors during build).
- **State Management:** React state is synchronized directly with Supabase Realtime channels.
- **Design System:** Tailwind CSS with a custom dark theme ensures a professional, high-contrast interface suitable for industrial monitoring.
