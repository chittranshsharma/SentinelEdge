"""
SentinelEdge — Telegram Alert Client
======================================
Sends formatted anomaly alerts to a Telegram chat/channel.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


def format_fault_label(label: str) -> str:
    """Convert snake_case fault label to Title Case."""
    return label.replace("_", " ").title()


def format_alert_message(
    fault_label:          str,
    confidence:           float,
    timestamp_epoch:      int,
    gps_lat:              Optional[float],
    gps_lng:              Optional[float],
    accel_rms_x:          Optional[float],
    temperature:          Optional[float],
    humidity:             Optional[float],
    air_quality_raw:      Optional[int],
    inference_latency_ms: Optional[int],
    llm_explanation:      Optional[str] = None,
) -> str:
    """
    Build the Telegram message matching the spec format.
    Uses Telegram MarkdownV2 escaping for special characters.
    """
    try:
        ts_str = datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except Exception:
        ts_str = "Unknown time"

    if gps_lat is not None and gps_lng is not None:
        loc_str = f"{gps_lat:.4f}°N, {gps_lng:.4f}°E"
    else:
        loc_str = "No GPS fix"

    rms_str  = f"{accel_rms_x:.2f}g" if accel_rms_x is not None else "N/A"
    temp_str = f"{temperature:.1f}°C" if temperature is not None else "N/A"
    hum_str  = f"{humidity:.0f}%"     if humidity is not None else "N/A"
    aq_str   = str(air_quality_raw)   if air_quality_raw is not None else "N/A"
    lat_ms   = f"{inference_latency_ms}ms" if inference_latency_ms else "N/A"

    explanation_block = ""
    if llm_explanation:
        explanation_block = f"\n🤖 AI Analysis:\n{llm_explanation}\n"

    message = (
        f"🚨 SENTINELEDGE ALERT\n\n"
        f"Fault: {format_fault_label(fault_label)} ({confidence*100:.0f}% confidence)\n"
        f"Time: {ts_str}\n"
        f"Location: {loc_str}\n"
        f"\n📊 Sensors:\n"
        f"• Vibration RMS: {rms_str}\n"
        f"• Temp: {temp_str} | Humidity: {hum_str}\n"
        f"• Air Quality: {aq_str} ADC\n"
        f"{explanation_block}"
        f"\n⚡ Inference latency: {lat_ms} (on-device)"
    )
    return message


async def send_telegram_alert(message: str) -> bool:
    """
    Send a message to the configured Telegram chat.
    Returns True if successful.
    """
    url = f"{TELEGRAM_API_BASE}/bot{settings.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id":    settings.telegram_chat_id,
        "text":       message,
        "parse_mode": "",  # Plain text to avoid MarkdownV2 escaping issues
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Telegram alert sent successfully")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Telegram HTTP error {e.response.status_code}: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


async def send_anomaly_alert(
    fault_label:          str,
    confidence:           float,
    timestamp_epoch:      int,
    gps_lat:              Optional[float] = None,
    gps_lng:              Optional[float] = None,
    accel_rms_x:          Optional[float] = None,
    temperature:          Optional[float] = None,
    humidity:             Optional[float] = None,
    air_quality_raw:      Optional[int]   = None,
    inference_latency_ms: Optional[int]   = None,
    llm_explanation:      Optional[str]   = None,
) -> tuple[bool, str]:
    """
    Format and send a complete anomaly alert.
    Returns (success, formatted_message) tuple.
    The message is returned so it can be stored in the alerts table.
    """
    message = format_alert_message(
        fault_label, confidence, timestamp_epoch,
        gps_lat, gps_lng, accel_rms_x,
        temperature, humidity, air_quality_raw,
        inference_latency_ms, llm_explanation,
    )
    success = await send_telegram_alert(message)
    return success, message
