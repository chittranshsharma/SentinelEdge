// SentinelEdge — mqtt_handler.cpp
// =================================
// WiFi + MQTT connection management and JSON payload publishing.

#include "mqtt_handler.h"

// ── Module-level state ─────────────────────────────────────────────────────────
static WiFiClient   wifiClient;
static PubSubClient mqttClient(wifiClient);
static const char*  _deviceId  = nullptr;
static const char*  _mqttHost  = nullptr;
static uint16_t     _mqttPort  = 1883;

// Reconnect backoff state
static unsigned long lastReconnectAttempt = 0;
static const unsigned long RECONNECT_INTERVAL_MS = 5000;

// Fault class label strings
static const char* FAULT_LABELS[] = {
    "normal",
    "imbalance",
    "obstruction",
    "loose_mount"
};

// ── WiFi ───────────────────────────────────────────────────────────────────────
static void wifiConnect(const char* ssid, const char* password) {
    Serial.printf("[WiFi] Connecting to %s", ssid);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected. IP: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("[WiFi] RSSI: %d dBm\n", WiFi.RSSI());
    } else {
        Serial.println("\n[WiFi] Connection FAILED — continuing without network");
        // Device continues operating locally; anomaly detection still works.
        // MQTT publishes will fail silently until WiFi recovers.
    }
}

// ── MQTT Reconnect ─────────────────────────────────────────────────────────────
static bool mqttReconnect() {
    if (mqttClient.connected()) return true;

    unsigned long now = millis();
    if (now - lastReconnectAttempt < RECONNECT_INTERVAL_MS) return false;
    lastReconnectAttempt = now;

    Serial.printf("[MQTT] Connecting to %s:%d as %s...", _mqttHost, _mqttPort, _deviceId);

    // LWT: if we disconnect ungracefully, broker publishes "offline" to status topic
    bool ok = mqttClient.connect(
        _deviceId,                  // client ID
        nullptr,                    // username (none)
        nullptr,                    // password (none)
        TOPIC_STATUS,               // LWT topic
        0,                          // LWT QoS
        true,                       // LWT retain
        "{\"status\":\"offline\"}"  // LWT payload
    );

    if (ok) {
        Serial.println(" connected");
        // Announce online status
        mqttClient.publish(TOPIC_STATUS,
            "{\"status\":\"online\"}", true);  // retain=true
        return true;
    } else {
        Serial.printf(" failed (state=%d)\n", mqttClient.state());
        return false;
    }
}

// ── Initialization ─────────────────────────────────────────────────────────────
void mqttInit(
    const char* ssid,
    const char* password,
    const char* mqttHost,
    uint16_t    mqttPort,
    const char* deviceId
) {
    _deviceId = deviceId;
    _mqttHost = mqttHost;
    _mqttPort = mqttPort;

    wifiConnect(ssid, password);

    mqttClient.setServer(mqttHost, mqttPort);
    // Raise payload buffer from PubSubClient default (256) to 1024 bytes
    mqttClient.setBufferSize(1024);
    // Keep-alive: 60 seconds
    mqttClient.setKeepAlive(60);

    mqttReconnect();
}

// ── Loop ───────────────────────────────────────────────────────────────────────
bool mqttLoop() {
    if (!mqttClient.connected()) {
        mqttReconnect();
    }
    mqttClient.loop();
    return mqttClient.connected();
}

bool mqttConnected() {
    return mqttClient.connected();
}

// ── JSON Builder ───────────────────────────────────────────────────────────────
static void buildSensorsObject(JsonObject sensorsObj, const SensorSnapshot& s) {
    sensorsObj["accel_rms_x"]    = serialized(String(s.accelRmsX, 3));
    sensorsObj["accel_rms_y"]    = serialized(String(s.accelRmsY, 3));
    sensorsObj["accel_rms_z"]    = serialized(String(s.accelRmsZ, 3));
    sensorsObj["temperature"]    = serialized(String(s.temperature, 1));
    sensorsObj["humidity"]       = serialized(String(s.humidity, 1));
    sensorsObj["air_quality_raw"] = s.airQualityRaw;
    sensorsObj["gps_lat"]        = serialized(String(s.gpsLat, 6));
    sensorsObj["gps_lng"]        = serialized(String(s.gpsLng, 6));
    sensorsObj["gps_fix"]        = s.gpsFix;
}

// ── Publish Anomaly ────────────────────────────────────────────────────────────
void publishAnomaly(
    int                  faultClass,
    float                confidence,
    unsigned long        inferenceLatencyMs,
    const SensorSnapshot& sensors
) {
    if (!mqttClient.connected()) {
        Serial.println("[MQTT] Cannot publish anomaly — not connected");
        return;
    }

    // JSON payload (target < 512 bytes, well within 1KB buffer)
    // Estimated size: ~350 bytes
    JsonDocument doc;
    doc["device_id"]            = _deviceId;
    doc["timestamp"]            = sensors.timestampEpoch;
    doc["fault_class"]          = faultClass;
    doc["fault_label"]          = FAULT_LABELS[faultClass];
    doc["confidence"]           = serialized(String(confidence, 3));
    doc["inference_latency_ms"] = inferenceLatencyMs;

    JsonObject s = doc["sensors"].to<JsonObject>();
    buildSensorsObject(s, sensors);

    char payload[768];
    size_t len = serializeJson(doc, payload, sizeof(payload));

    bool ok = mqttClient.publish(TOPIC_ANOMALY, payload, false);  // retain=false
    if (ok) {
        Serial.printf("[MQTT] Anomaly published: %s (%.0f%%) — %zu bytes\n",
            FAULT_LABELS[faultClass], confidence * 100.0f, len);
    } else {
        Serial.printf("[MQTT] Anomaly publish FAILED (payload=%zu bytes)\n", len);
    }
}

// ── Publish Heartbeat ──────────────────────────────────────────────────────────
void publishHeartbeat(const SensorSnapshot& sensors) {
    if (!mqttClient.connected()) return;

    JsonDocument doc;
    doc["device_id"]  = _deviceId;
    doc["timestamp"]  = sensors.timestampEpoch;
    doc["uptime_ms"]  = millis();
    doc["free_heap"]  = ESP.getFreeHeap();

    JsonObject s = doc["sensors"].to<JsonObject>();
    buildSensorsObject(s, sensors);

    char payload[512];
    size_t len = serializeJson(doc, payload, sizeof(payload));

    mqttClient.publish(TOPIC_HEARTBEAT, payload, false);
    // Not logging to Serial to avoid 5s spam — uncomment for debugging:
    // Serial.printf("[MQTT] Heartbeat: %zu bytes\n", len);
}
