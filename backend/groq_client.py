"""
SentinelEdge — Groq LLM Explanation Client
============================================
Generates 2-sentence fault explanations using Groq API (llama-3.3-70b-versatile).

Prompt is designed to produce specific, actionable maintenance guidance —
NOT generic "consult a professional" responses.
"""

import logging
from typing import Optional

from groq import AsyncGroq
from config import settings

logger = logging.getLogger(__name__)

# Groq async client (singleton)
_groq_client: Optional[AsyncGroq] = None


def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
        logger.info("Groq client initialized")
    return _groq_client


async def generate_explanation(
    fault_label:     str,
    confidence:      float,
    accel_rms_x:     Optional[float],
    temperature:     Optional[float],
    humidity:        Optional[float],
    air_quality_raw: Optional[int],
    gps_lat:         Optional[float],
    gps_lng:         Optional[float],
    timestamp:       int,
) -> Optional[str]:
    """
    Call Groq LLM to generate a 2-sentence fault explanation.

    Returns the explanation string or None if the API call fails.
    Designed to not block the main anomaly pipeline — caller should
    run this as an async task and update the DB record when done.
    """
    # Format optional values for the prompt
    rms_str   = f"{accel_rms_x:.2f}g" if accel_rms_x is not None else "N/A"
    temp_str  = f"{temperature:.1f}°C" if temperature is not None else "N/A"
    hum_str   = f"{humidity:.0f}%"     if humidity is not None else "N/A"
    aq_str    = str(air_quality_raw)   if air_quality_raw is not None else "N/A"
    loc_str   = (f"{gps_lat:.4f}, {gps_lng:.4f}"
                 if gps_lat is not None and gps_lng is not None
                 else "unavailable")

    # Convert epoch to readable time
    from datetime import datetime, timezone
    try:
        ts_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except Exception:
        ts_str = f"epoch {timestamp}"

    prompt = f"""You are an industrial equipment monitoring assistant.

An anomaly was detected by an edge sensor node. Here is the context:

Fault classification: {fault_label} (confidence: {confidence*100:.0f}%)
Vibration RMS (x-axis): {rms_str}
Temperature: {temp_str}
Humidity: {hum_str}
Air quality (raw ADC): {aq_str}
Location: {loc_str}
Time: {ts_str}

In exactly 2 sentences:
1. Explain the likely physical cause of this fault based on the vibration signature.
2. State the recommended immediate action.

Be specific. Do not be generic. Do not say "consult a professional."
Do not number your sentences. Write them as a continuous paragraph."""

    try:
        client  = get_groq_client()
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.groq_max_tokens,
            temperature=0.3,  # Low temperature for consistent, factual responses
        )
        explanation = response.choices[0].message.content.strip()
        logger.info(f"Groq explanation generated ({len(explanation)} chars)")
        return explanation
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return None
