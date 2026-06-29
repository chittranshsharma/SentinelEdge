// SentinelEdge — mqtt_handler.h
// ==============================
// WiFi connection and MQTT publish/subscribe for ESP32.
//
// Topics:
//   sentineledge/anomaly    — fault event (confidence > threshold, class != normal)
//   sentineledge/heartbeat  — periodic sensor readings (every 5 seconds)
//   sentineledge/status     — online/offline (LWT = Last Will and Testament)

#ifndef MQTT_HANDLER_H
#define MQTT_HANDLER_H

#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <ArduinoJson.h>

// ── Topics ─────────────────────────────────────────────────────────────────────
#define TOPIC_ANOMALY    "sentineledge/anomaly"
#define TOPIC_HEARTBEAT  "sentineledge/heartbeat"
#define TOPIC_STATUS     "sentineledge/status"

// ── Sensor Data Snapshot ────────────────────────────────────────────────────────
// Passed to publishAnomaly() and publishHeartbeat().
// GPS values use last known fix (may be stale if indoor with no satellite lock).
struct SensorSnapshot {
    float accelRmsX;
    float accelRmsY;
    float accelRmsZ;
    float temperature;
    float humidity;
    int   airQualityRaw;
    double gpsLat;
    double gpsLng;
    bool  gpsFix;
    unsigned long timestampEpoch;  // Unix timestamp (from GPS or millis() fallback)
};

// ── Function Declarations ──────────────────────────────────────────────────────

/**
 * Initialize WiFi and MQTT client.
 * Blocks until WiFi is connected (with retries).
 *
 * @param ssid         WiFi network name
 * @param password     WiFi password
 * @param mqttHost     MQTT broker hostname or IP (Raspberry Pi address)
 * @param mqttPort     MQTT broker port (default: 1883)
 * @param deviceId     Unique device identifier (e.g., "sentineledge-001")
 */
void mqttInit(
    const char* ssid,
    const char* password,
    const char* mqttHost,
    uint16_t    mqttPort,
    const char* deviceId
);

/**
 * Maintain MQTT connection. Call every loop iteration.
 * Reconnects automatically if connection drops.
 * Returns true if connected.
 */
bool mqttLoop();

/**
 * Publish an anomaly event to sentineledge/anomaly.
 *
 * Only called when:
 *   - faultClass != kClassNormal (0)
 *   - confidence > kConfidenceThreshold (0.75)
 *
 * Payload: JSON object with fault classification, sensor readings, GPS location.
 * Max payload size: ~512 bytes (well under MQTT default 256-byte limit of PubSubClient;
 * setBufferSize(1024) is called in mqttInit to raise the limit).
 *
 * @param faultClass         Predicted class index (1=imbalance, 2=obstruction, 3=loose)
 * @param confidence         Softmax probability of predicted class (0.0–1.0)
 * @param inferenceLatencyMs Total pipeline time: feature extraction + inference
 * @param sensors            All sensor readings at time of anomaly
 */
void publishAnomaly(
    int                  faultClass,
    float                confidence,
    unsigned long        inferenceLatencyMs,
    const SensorSnapshot& sensors
);

/**
 * Publish periodic heartbeat to sentineledge/heartbeat (every 5 seconds).
 * Sent regardless of anomaly detection — used for dashboard "last seen" indicator.
 *
 * @param sensors  Current sensor readings
 */
void publishHeartbeat(const SensorSnapshot& sensors, int currentState);

/**
 * Returns true if MQTT client is currently connected.
 */
bool mqttConnected();

#endif  // MQTT_HANDLER_H
