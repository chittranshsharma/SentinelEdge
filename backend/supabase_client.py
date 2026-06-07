"""
SentinelEdge — Supabase Client
================================
Writes sensor readings, anomalies, and alerts to Supabase.
Uses supabase-py with service key for full database access.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client, Client
from config import settings
from models import SensorReadingRecord, AnomalyRecord

logger = logging.getLogger(__name__)

# ── Supabase Client (singleton) ────────────────────────────────────────────────
_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
        logger.info("Supabase client initialized")
    return _client


def epoch_to_datetime(epoch_seconds: int) -> str:
    """Convert Unix epoch seconds to ISO 8601 string for Supabase timestamp columns."""
    try:
        dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        return dt.isoformat()
    except (OSError, OverflowError, ValueError):
        # Fallback: use current time if epoch is invalid
        return datetime.now(timezone.utc).isoformat()


# ── sensor_readings ────────────────────────────────────────────────────────────

async def write_sensor_reading(
    device_id: str,
    timestamp_epoch: int,
    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    air_quality_raw: Optional[int] = None,
    accel_rms_x: Optional[float] = None,
    accel_rms_y: Optional[float] = None,
    accel_rms_z: Optional[float] = None,
    gps_lat: Optional[float] = None,
    gps_lng: Optional[float] = None,
    gps_fix: bool = False,
) -> Optional[str]:
    """
    Insert a row into sensor_readings.
    Called for both heartbeat and anomaly events.
    Returns the inserted row's id or None on failure.
    """
    try:
        client = get_client()
        record = {
            "device_id":       device_id,
            "timestamp":       epoch_to_datetime(timestamp_epoch),
            "temperature":     temperature,
            "humidity":        humidity,
            "air_quality_raw": air_quality_raw,
            "accel_rms_x":     accel_rms_x,
            "accel_rms_y":     accel_rms_y,
            "accel_rms_z":     accel_rms_z,
            "gps_lat":         gps_lat,
            "gps_lng":         gps_lng,
            "gps_fix":         gps_fix,
        }
        result = client.table("sensor_readings").insert(record).execute()
        row_id = result.data[0]["id"] if result.data else None
        logger.debug(f"sensor_reading inserted: {row_id}")
        return row_id
    except Exception as e:
        logger.error(f"Failed to write sensor_reading: {e}")
        return None


# ── anomalies ─────────────────────────────────────────────────────────────────

async def write_anomaly(
    device_id: str,
    timestamp_epoch: int,
    fault_class: int,
    fault_label: str,
    confidence: float,
    inference_latency_ms: Optional[int] = None,
    accel_rms_x: Optional[float] = None,
    accel_rms_y: Optional[float] = None,
    accel_rms_z: Optional[float] = None,
    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    air_quality_raw: Optional[int] = None,
    gps_lat: Optional[float] = None,
    gps_lng: Optional[float] = None,
    llm_explanation: Optional[str] = None,
    telegram_sent: bool = False,
) -> Optional[str]:
    """
    Insert a row into anomalies table.
    Returns the new anomaly's UUID or None on failure.
    """
    try:
        client = get_client()
        record = {
            "device_id":            device_id,
            "timestamp":            epoch_to_datetime(timestamp_epoch),
            "fault_class":          fault_class,
            "fault_label":          fault_label,
            "confidence":           confidence,
            "inference_latency_ms": inference_latency_ms,
            "accel_rms_x":          accel_rms_x,
            "accel_rms_y":          accel_rms_y,
            "accel_rms_z":          accel_rms_z,
            "temperature":          temperature,
            "humidity":             humidity,
            "air_quality_raw":      air_quality_raw,
            "gps_lat":              gps_lat,
            "gps_lng":              gps_lng,
            "llm_explanation":      llm_explanation,
            "telegram_sent":        telegram_sent,
        }
        result = client.table("anomalies").insert(record).execute()
        row_id = result.data[0]["id"] if result.data else None
        logger.info(f"Anomaly inserted: {row_id} | {fault_label} ({confidence:.0%})")
        return row_id
    except Exception as e:
        logger.error(f"Failed to write anomaly: {e}")
        return None


async def update_anomaly_explanation(anomaly_id: str, llm_explanation: str) -> bool:
    """Update llm_explanation after Groq responds."""
    try:
        client = get_client()
        client.table("anomalies") \
            .update({"llm_explanation": llm_explanation}) \
            .eq("id", anomaly_id) \
            .execute()
        return True
    except Exception as e:
        logger.error(f"Failed to update anomaly explanation: {e}")
        return False


async def update_anomaly_telegram_sent(anomaly_id: str) -> bool:
    """Mark Telegram alert as sent."""
    try:
        client = get_client()
        client.table("anomalies") \
            .update({"telegram_sent": True}) \
            .eq("id", anomaly_id) \
            .execute()
        return True
    except Exception as e:
        logger.error(f"Failed to update telegram_sent: {e}")
        return False


# ── alerts ────────────────────────────────────────────────────────────────────

async def write_alert(
    anomaly_id: str,
    channel: str,
    message: str,
) -> Optional[str]:
    """Insert a row into the alerts log table."""
    try:
        client = get_client()
        record = {
            "anomaly_id": anomaly_id,
            "channel":    channel,
            "message":    message,
        }
        result = client.table("alerts").insert(record).execute()
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        logger.error(f"Failed to write alert: {e}")
        return None
