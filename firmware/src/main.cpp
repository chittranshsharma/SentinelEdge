// SentinelEdge — main.cpp
// =========================
// ESP32 edge AI firmware for vibration-based fault detection.
//
// System overview:
//   MPU6050 sampled at 100Hz → circular buffer (200 samples)
//   Every 100 samples: 42-feature extraction → normalize → int8 quantize
//   → TFLite Micro inference → if fault + confidence>75%: MQTT publish
//   Every 5s: heartbeat publish with all sensor readings
//
// Pin assignments (DO NOT CHANGE — matches hardware spec):
//   MPU6050: SDA=21, SCL=22   (I2C default on ESP32)
//   DHT11:   GPIO 4
//   MQ5:     GPIO 34           (ADC1_CH6 — input only pin, 12-bit ADC)
//   NEO-6M:  RX2=16, TX2=17   (UART2)
//
// Benchmark targets:
//   Feature extraction : < 30ms
//   TFLite inference   : < 70ms
//   Total pipeline     : < 100ms
//   SRAM for ML        : < 200KB of 520KB total

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>

// ── Sensors ──────────────────────────────────────────────────────────────────
#include <MPU6050.h>
#include <DHT.h>
#include <TinyGPSPlus.h>

// ── ML ───────────────────────────────────────────────────────────────────────
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model/fault_model.h"
#include "model/model_settings.h"

// ── Local modules ─────────────────────────────────────────────────────────────
#include "features.h"
#include "mqtt_handler.h"

// ─────────────────────────────────────────────────────────────────────────────
// USER CONFIGURATION — fill these before flashing
// ─────────────────────────────────────────────────────────────────────────────
static const char* WIFI_SSID     = "YOUR_WIFI_SSID";
static const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
static const char* MQTT_HOST     = "192.168.1.100";  // Raspberry Pi IP
static const uint16_t MQTT_PORT  = 1883;
static const char* DEVICE_ID     = "sentineledge-001";
// ─────────────────────────────────────────────────────────────────────────────

// ── Pin Definitions ───────────────────────────────────────────────────────────
#define DHT_PIN       4
#define DHT_TYPE      DHT11
#define MQ5_PIN       34   // ADC1_CH6
#define GPS_RX_PIN    16
#define GPS_TX_PIN    17
#define GPS_BAUD      9600

// ── Timing ────────────────────────────────────────────────────────────────────
#define SAMPLE_INTERVAL_US  10000UL  // 10ms = 100Hz
#define HEARTBEAT_INTERVAL_MS 5000UL
#define DHT_READ_INTERVAL_MS  2000UL  // DHT11 max sample rate = 1Hz, read every 2s

// MQ5 warmup: needs 3-5 minutes after power-on for stable readings.
// We log a warning for the first 5 minutes but don't block operation.
#define MQ5_WARMUP_MS (5UL * 60UL * 1000UL)

// ── TFLite Micro ──────────────────────────────────────────────────────────────
// Static tensor arena — MUST be static/global (no heap allocation in inference path)
static uint8_t tensorArena[kTensorArenaSize];

static const tflite::Model*        tflModel     = nullptr;
static tflite::MicroInterpreter*   tflInterp    = nullptr;
static TfLiteTensor*               tflInput     = nullptr;
static TfLiteTensor*               tflOutput    = nullptr;
static tflite::AllOpsResolver      tflResolver;

// ── Sensor Objects ────────────────────────────────────────────────────────────
static MPU6050  mpu;
static DHT      dht(DHT_PIN, DHT_TYPE);
static TinyGPSPlus gps;
static HardwareSerial gpsSerial(2);  // UART2: RX=16, TX=17

// ── Cached Sensor Values ──────────────────────────────────────────────────────
// DHT11 and GPS are read infrequently; values cached between reads
static float    cachedTemp        = 0.0f;
static float    cachedHumidity    = 0.0f;
static int      cachedAirQuality  = 0;
static double   cachedGpsLat      = 0.0;
static double   cachedGpsLng      = 0.0;
static bool     cachedGpsFix      = false;
static unsigned long cachedGpsAge = 0;  // millis() when GPS was last updated
static unsigned long lastDhtRead  = 0;
static unsigned long lastHeartbeat = 0;

// ── Inference state ───────────────────────────────────────────────────────────
static int           lastPredictedClass  = 0;
static float         lastConfidence      = 0.0f;
static unsigned long lastTotalLatencyMs  = 0;
static unsigned long lastFeatureUs       = 0;
static unsigned long lastInferenceUs     = 0;

// ─────────────────────────────────────────────────────────────────────────────
// TFLite Micro Setup
// ─────────────────────────────────────────────────────────────────────────────
static void setupTFLite() {
    Serial.println("[TFLite] Initializing...");

    tflModel = tflite::GetModel(fault_model_data);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[TFLite] ERROR: Model schema %d, expected %d\n",
            tflModel->version(), TFLITE_SCHEMA_VERSION);
        while (true) delay(1000);
    }

    // Use static placement (no heap) for interpreter
    static tflite::MicroInterpreter staticInterpreter(
        tflModel, tflResolver, tensorArena, kTensorArenaSize
    );
    tflInterp = &staticInterpreter;

    TfLiteStatus allocStatus = tflInterp->AllocateTensors();
    if (allocStatus != kTfLiteOk) {
        Serial.printf("[TFLite] ERROR: AllocateTensors() failed!\n");
        Serial.printf("         Increase kTensorArenaSize in model_settings.h\n");
        Serial.printf("         Current: %d bytes\n", kTensorArenaSize);
        while (true) delay(1000);
    }

    tflInput  = tflInterp->input(0);
    tflOutput = tflInterp->output(0);

    // Verify tensor shapes and types
    Serial.printf("[TFLite] Input  tensor: [%d, %d] %s\n",
        tflInput->dims->data[0], tflInput->dims->data[1],
        tflInput->type == kTfLiteInt8 ? "int8" : "WRONG TYPE");
    Serial.printf("[TFLite] Output tensor: [%d, %d] %s\n",
        tflOutput->dims->data[0], tflOutput->dims->data[1],
        tflOutput->type == kTfLiteInt8 ? "int8" : "WRONG TYPE");
    Serial.printf("[TFLite] Arena used   : %d bytes / %d bytes (%.1f%%)\n",
        tflInterp->arena_used_bytes(), kTensorArenaSize,
        (float)tflInterp->arena_used_bytes() / kTensorArenaSize * 100.0f);
    Serial.printf("[TFLite] Model size   : %u bytes in flash\n", kModelSizeBytes);
    Serial.printf("[TFLite] Free heap    : %d bytes\n", ESP.getFreeHeap());
    Serial.println("[TFLite] Ready");
}

// ─────────────────────────────────────────────────────────────────────────────
// Sensor Setup
// ─────────────────────────────────────────────────────────────────────────────
static void setupSensors() {
    // MPU6050 — I2C
    Wire.begin(21, 22);  // SDA=21, SCL=22
    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("[MPU6050] ERROR: Not found on I2C!");
        // This is fatal — MPU6050 is the only ML input source
        while (true) delay(1000);
    }
    // Set sensitivity: ±2g accel, ±250°/s gyro (highest precision)
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    // Disable sleep mode
    mpu.setSleepEnabled(false);
    Serial.println("[MPU6050] OK — ±2g / ±250dps");

    // DHT11 — temperature/humidity
    dht.begin();
    Serial.println("[DHT11]   OK — GPIO4");

    // MQ5 — gas/air quality (ADC)
    // NOTE: MQ5 needs 3-5 minutes warmup after power-on for stable readings.
    // Raw ADC value is 0-4095 (12-bit). Meaningful comparison requires calibration.
    analogReadResolution(12);
    analogSetPinAttenuation(MQ5_PIN, ADC_11db);  // 0-3.9V range
    Serial.printf("[MQ5]     OK — GPIO%d (ADC 12-bit, 3-5 min warmup needed)\n", MQ5_PIN);

    // NEO-6M GPS — UART2
    gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
    Serial.printf("[GPS]     OK — UART2 RX=%d TX=%d\n", GPS_RX_PIN, GPS_TX_PIN);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sensor Reads (cached / rate-limited)
// ─────────────────────────────────────────────────────────────────────────────
static void readDHT() {
    // DHT11 max reliable rate: 1 reading per second. Read every 2s.
    unsigned long now = millis();
    if (now - lastDhtRead < DHT_READ_INTERVAL_MS) return;
    lastDhtRead = now;

    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t) && !isnan(h)) {
        cachedTemp     = t;
        cachedHumidity = h;
    }
    // If DHT11 read fails (returns NaN), keep previous cached value
}

static void readMQ5() {
    // Raw ADC — no conversion needed (backend receives raw ADC value)
    cachedAirQuality = analogRead(MQ5_PIN);
    if (millis() < MQ5_WARMUP_MS) {
        // MQ5 still warming up — value is unreliable
        // Log to help the dashboard know reading is not yet stable
    }
}

static void readGPS() {
    // Drain UART2 buffer; TinyGPSPlus parses NMEA sentences
    while (gpsSerial.available()) {
        gps.encode(gpsSerial.read());
    }
    if (gps.location.isValid() && gps.location.isUpdated()) {
        cachedGpsLat  = gps.location.lat();
        cachedGpsLng  = gps.location.lng();
        cachedGpsFix  = true;
        cachedGpsAge  = millis();
    }
    // Indoor / no satellite: cachedGpsFix remains false,
    // last known coordinates are used (or 0.0, 0.0 if never had fix)
}

// ─────────────────────────────────────────────────────────────────────────────
// Build SensorSnapshot from cached values
// ─────────────────────────────────────────────────────────────────────────────
static SensorSnapshot buildSnapshot() {
    SensorSnapshot s;
    // Accel RMS comes from feature array (features[3] = ax_rms, etc.)
    // Feature order: [ax_mean, ax_std, ax_variance, ax_rms, ...] per axis
    // ax_rms = features[3], ay_rms = features[10], az_rms = features[17]
    s.accelRmsX      = features[3];   // ax_rms (raw, before normalization)
    s.accelRmsY      = features[10];  // ay_rms
    s.accelRmsZ      = features[17];  // az_rms
    s.temperature    = cachedTemp;
    s.humidity       = cachedHumidity;
    s.airQualityRaw  = cachedAirQuality;
    s.gpsLat         = cachedGpsLat;
    s.gpsLng         = cachedGpsLng;
    s.gpsFix         = cachedGpsFix;

    // Timestamp: use GPS time if available, else millis()-based approximation
    if (gps.time.isValid()) {
        // GPS provides UTC time — construct approximate epoch
        // (simplified: seconds since midnight; add date offset if needed)
        s.timestampEpoch = (unsigned long)gps.time.second()
                         + (unsigned long)gps.time.minute() * 60
                         + (unsigned long)gps.time.hour()   * 3600;
    } else {
        s.timestampEpoch = millis() / 1000;  // uptime in seconds as fallback
    }

    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// TFLite Inference Pipeline
// ─────────────────────────────────────────────────────────────────────────────
static void runInference() {
    unsigned long pipelineStart = millis();

    // 1. Feature extraction (reads circular buffer, computes 42 features)
    unsigned long t0 = micros();
    extractFeatures();   // writes raw features to features[]
    lastFeatureUs = micros() - t0;

    // 2. Normalize features (subtract mean, divide by scale)
    // IMPORTANT: normalizeFeatures() overwrites features[] in place.
    // If you need raw RMS for the MQTT payload, read it BEFORE normalization.
    SensorSnapshot snapshot = buildSnapshot();  // reads raw features[3,10,17]
    normalizeFeatures();

    // 3. Quantize normalized features → int8 → TFLite input tensor
    quantizeFeatures(tflInput->data.int8);

    // 4. Run inference
    t0 = micros();
    TfLiteStatus status = tflInterp->Invoke();
    lastInferenceUs = micros() - t0;

    if (status != kTfLiteOk) {
        Serial.println("[TFLite] ERROR: Invoke() failed!");
        return;
    }

    // 5. Argmax on int8 output (no dequantization needed for class selection)
    int8_t* output = tflOutput->data.int8;
    int  bestClass  = 0;
    int8_t bestVal  = output[0];
    for (int i = 1; i < kNumClasses; i++) {
        if (output[i] > bestVal) {
            bestVal   = output[i];
            bestClass = i;
        }
    }

    // 6. Dequantize winning class for confidence score
    float confidence = ((float)bestVal - kOutputZeroPoint) * kOutputScale;
    // Clamp to [0, 1] (dequant can occasionally exceed due to quantization error)
    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;

    lastPredictedClass = bestClass;
    lastConfidence     = confidence;
    lastTotalLatencyMs = millis() - pipelineStart;

    // 7. Log inference result
    Serial.printf("[INF] %s (%.0f%%)  |  feat=%lums  infer=%lums  total=%lums  free=%d\n",
        kFaultLabels[bestClass],
        confidence * 100.0f,
        lastFeatureUs / 1000,
        lastInferenceUs / 1000,
        lastTotalLatencyMs,
        ESP.getFreeHeap()
    );

    // 8. Publish anomaly if fault detected with sufficient confidence
    if (bestClass != kClassNormal && confidence >= kConfidenceThreshold) {
        readDHT();
        readMQ5();
        readGPS();
        snapshot = buildSnapshot();  // refresh with latest sensor reads

        publishAnomaly(bestClass, confidence, lastTotalLatencyMs, snapshot);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Collection Mode (ENV_COLLECTION)
// Prints raw MPU6050 CSV at 100Hz — no TFLite, no MQTT
// ─────────────────────────────────────────────────────────────────────────────
#ifdef ENV_COLLECTION
static void dataCollectionLoop() {
    static unsigned long lastSampleUs = 0;
    unsigned long now = micros();
    if (now - lastSampleUs < SAMPLE_INTERVAL_US) return;
    lastSampleUs = now;

    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert raw to physical units
    // Accel sensitivity: ±2g range → 16384 LSB/g → divide by 16384 * 9.81
    const float ACCEL_SCALE = 9.81f / 16384.0f;
    // Gyro sensitivity: ±250°/s range → 131 LSB/deg/s
    const float GYRO_SCALE  = 1.0f / 131.0f;

    Serial.printf("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        ax * ACCEL_SCALE, ay * ACCEL_SCALE, az * ACCEL_SCALE,
        gx * GYRO_SCALE,  gy * GYRO_SCALE,  gz * GYRO_SCALE
    );
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("\n╔══════════════════════════════════╗");
    Serial.println("║  SentinelEdge  —  Edge AI Node  ║");
    Serial.println("╚══════════════════════════════════╝");
    Serial.printf("  SDK version : %s\n", ESP.getSdkVersion());
    Serial.printf("  Free heap   : %d bytes\n", ESP.getFreeHeap());
    Serial.printf("  CPU freq    : %d MHz\n", getCpuFrequencyMhz());
    Serial.println();

    setupSensors();

#ifdef ENV_INFERENCE
    initBuffers();
    setupTFLite();
    mqttInit(WIFI_SSID, WIFI_PASSWORD, MQTT_HOST, MQTT_PORT, DEVICE_ID);
    Serial.println("\n[SentinelEdge] Inference mode — sampling at 100Hz");
#endif

#ifdef ENV_COLLECTION
    Serial.println("[SentinelEdge] Data collection mode — printing CSV at 100Hz");
    Serial.println("# ax,ay,az,gx,gy,gz");  // header comment for logger
#endif

    lastHeartbeat = millis();
    lastDhtRead   = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Loop
// ─────────────────────────────────────────────────────────────────────────────
void loop() {

#ifdef ENV_COLLECTION
    dataCollectionLoop();
    return;
#endif

#ifdef ENV_INFERENCE
    // ── 100Hz MPU6050 sampling ─────────────────────────────────────────────
    static unsigned long lastSampleUs = 0;
    unsigned long nowUs = micros();

    if (nowUs - lastSampleUs >= SAMPLE_INTERVAL_US) {
        lastSampleUs = nowUs;

        // Read MPU6050 raw 16-bit values
        int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;
        mpu.getMotion6(&rawAx, &rawAy, &rawAz, &rawGx, &rawGy, &rawGz);

        // Convert to physical units (matching Python training data units)
        // Accel: ±2g → 16384 LSB/g → multiply by g/16384 = 9.81/16384
        // Gyro:  ±250°/s → 131 LSB/(°/s)
        const float ACCEL_SCALE = 9.81f / 16384.0f;
        const float GYRO_SCALE  = 1.0f  / 131.0f;

        float ax = rawAx * ACCEL_SCALE;
        float ay = rawAy * ACCEL_SCALE;
        float az = rawAz * ACCEL_SCALE;
        float gx = rawGx * GYRO_SCALE;
        float gy = rawGy * GYRO_SCALE;
        float gz = rawGz * GYRO_SCALE;

        pushSample(ax, ay, az, gx, gy, gz);

        // ── Feature extraction + inference (every STEP_SIZE=100 samples) ──
        if (windowReady()) {
            runInference();
        }
    }

    // ── GPS parsing (continuous, non-blocking) ─────────────────────────────
    readGPS();

    // ── Heartbeat publish (every 5 seconds) ───────────────────────────────
    unsigned long nowMs = millis();
    if (nowMs - lastHeartbeat >= HEARTBEAT_INTERVAL_MS) {
        lastHeartbeat = nowMs;
        readDHT();
        readMQ5();
        SensorSnapshot snapshot = buildSnapshot();
        publishHeartbeat(snapshot);
    }

    // ── MQTT keepalive ─────────────────────────────────────────────────────
    mqttLoop();

    // ── Drift check mode (send 'd' via serial to trigger) ─────────────────
#ifdef DRIFT_CHECK_ENABLED
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'd' || c == 'D') {
            Serial.println("[DRIFT] Starting feature drift check...");
            printDriftCheckFeatures();
        }
    }
#endif

#endif  // ENV_INFERENCE
}
