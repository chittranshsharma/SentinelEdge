"""
SentinelEdge — MQTT Client
============================
Subscribes to sentineledge/# and wires MQTT messages to:
  1. Supabase (sensor_readings + anomalies + alerts tables)
  2. Groq API (LLM explanation generation, async)
  3. Telegram (alert delivery)

Message flow:
  MQTT message received
    → Parse JSON → Pydantic model validation
    → Write sensor_reading (always)
    → If anomaly topic:
        a. Write anomaly record (without explanation yet)
        b. Spawn async task: Groq → update anomaly → Telegram → write alert
"""

import asyncio
import json
import logging
from datetime import datetime

import paho.mqtt.client as mqtt_paho

from config import settings
from models import AnomalyPayload, HeartbeatPayload
import supabase_client as db
import groq_client
import telegram_client

logger = logging.getLogger(__name__)

# ── MQTT Topics ────────────────────────────────────────────────────────────────
TOPIC_ANOMALY   = "sentineledge/anomaly"
TOPIC_HEARTBEAT = "sentineledge/heartbeat"
TOPIC_STATUS    = "sentineledge/status"
TOPIC_ALL       = "sentineledge/#"

# ── Async event loop (shared with FastAPI) ─────────────────────────────────────
_event_loop: asyncio.AbstractEventLoop = None


def set_event_loop(loop: asyncio.AbstractEventLoop):
    """Called from main.py lifespan to share the FastAPI event loop."""
    global _event_loop
    _event_loop = loop


# ── Anomaly Processing Pipeline ────────────────────────────────────────────────

async def process_anomaly(payload: AnomalyPayload):
    """
    Full anomaly processing pipeline:
    1. Write to anomalies table (immediately)
    2. Write to sensor_readings table
    3. Generate Groq explanation (async, ~1-2s)
    4. Update anomaly record with explanation
    5. Send Telegram alert
    6. Write alert log

    Runs as an asyncio task — does not block MQTT message processing.
    """
    s = payload.sensors

    # Step 1: Write sensor reading (always logged, even for anomalies)
    await db.write_sensor_reading(
        device_id=payload.device_id,
        timestamp_epoch=payload.timestamp,
        temperature=s.temperature,
        humidity=s.humidity,
        air_quality_raw=s.air_quality_raw,
        accel_rms_x=s.accel_rms_x,
        accel_rms_y=s.accel_rms_y,
        accel_rms_z=s.accel_rms_z,
        gps_lat=s.gps_lat,
        gps_lng=s.gps_lng,
        gps_fix=s.gps_fix or False,
        gps_satellites=s.gps_satellites,
        gps_simulated=s.gps_simulated or False,
    )

    # Step 2: Write anomaly record (initially without LLM explanation)
    anomaly_id = await db.write_anomaly(
        device_id=payload.device_id,
        timestamp_epoch=payload.timestamp,
        fault_class=payload.fault_class,
        fault_label=payload.fault_label,
        confidence=payload.confidence,
        inference_latency_ms=payload.inference_latency_ms,
        accel_rms_x=s.accel_rms_x,
        accel_rms_y=s.accel_rms_y,
        accel_rms_z=s.accel_rms_z,
        temperature=s.temperature,
        humidity=s.humidity,
        air_quality_raw=s.air_quality_raw,
        gps_lat=s.gps_lat,
        gps_lng=s.gps_lng,
        llm_explanation=None,
        telegram_sent=False,
    )

    if anomaly_id is None:
        logger.error("Failed to insert anomaly — skipping LLM + Telegram")
        return
    # Phase 1: Skip LLM and Telegram. Just ensure reliable data flow to Supabase.
    logger.info(f"Successfully processed transition: {payload.fault_label} (anomaly_id={anomaly_id})")


async def process_heartbeat(payload: HeartbeatPayload):
    """Write heartbeat sensor readings to sensor_readings table."""
    s = payload.sensors
    await db.write_sensor_reading(
        device_id=payload.device_id,
        timestamp_epoch=payload.timestamp,
        temperature=s.temperature,
        humidity=s.humidity,
        air_quality_raw=s.air_quality_raw,
        accel_rms_x=s.accel_rms_x,
        accel_rms_y=s.accel_rms_y,
        accel_rms_z=s.accel_rms_z,
        gps_lat=s.gps_lat,
        gps_lng=s.gps_lng,
        gps_fix=s.gps_fix or False,
        gps_satellites=s.gps_satellites,
        gps_simulated=s.gps_simulated or False,
        dht_success=payload.dht_success,
        dht_failure=payload.dht_failure,
        motion_state=payload.current_state,
        confidence=payload.confidence,
        unknown_candidate=payload.unknown_candidate or False,
        classification_margin=payload.classification_margin,
    )
    logger.debug(f"Heartbeat logged for {payload.device_id}")


# ── MQTT Callbacks ─────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info(f"Connected to MQTT broker at {settings.mqtt_broker_host}:{settings.mqtt_broker_port}")
        client.subscribe(TOPIC_ALL, qos=1)
        logger.info(f"Subscribed to {TOPIC_ALL}")
    else:
        logger.error(f"MQTT connection failed: rc={rc}")


def on_disconnect(client, userdata, rc, properties=None, reason_code=None):
    logger.warning(f"MQTT disconnected (rc={rc}) — will auto-reconnect")


def on_message(client, userdata, msg):
    """
    Called on every MQTT message.
    Runs in paho's background thread — schedule async tasks via event loop.
    """
    topic   = msg.topic
    payload = msg.payload.decode('utf-8', errors='ignore')

    logger.debug(f"MQTT message: {topic} ({len(payload)} bytes)")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON on {topic}: {e}")
        return

    if _event_loop is None:
        logger.error("Event loop not set — cannot process MQTT message")
        return

    if topic == TOPIC_ANOMALY:
        try:
            anomaly = AnomalyPayload(**data)
            asyncio.run_coroutine_threadsafe(
                process_anomaly(anomaly), _event_loop
            )
        except Exception as e:
            logger.error(f"Anomaly parse error: {e}\nPayload: {payload[:200]}")

    elif topic == TOPIC_HEARTBEAT:
        try:
            heartbeat = HeartbeatPayload(**data)
            asyncio.run_coroutine_threadsafe(
                process_heartbeat(heartbeat), _event_loop
            )
        except Exception as e:
            logger.error(f"Heartbeat parse error: {e}")

    elif topic == TOPIC_STATUS:
        logger.info(f"Device status: {payload}")


# ── MQTT Client Setup ──────────────────────────────────────────────────────────

def create_mqtt_client() -> mqtt_paho.Client:
    """Create and configure the MQTT client."""
    client = mqtt_paho.Client(
        client_id=settings.mqtt_client_id,
        callback_api_version=mqtt_paho.CallbackAPIVersion.VERSION2,
    )
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message

    # Auto-reconnect: try every 5 seconds
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    return client


def start_mqtt(loop: asyncio.AbstractEventLoop):
    """
    Connect to broker and start background network loop.
    loop_start() creates a daemon thread — runs alongside FastAPI.
    """
    set_event_loop(loop)

    client = create_mqtt_client()
    client.connect(
        host=settings.mqtt_broker_host,
        port=settings.mqtt_broker_port,
        keepalive=60,
    )
    client.loop_start()  # Non-blocking: starts background thread
    logger.info("MQTT background loop started")
    return client
