"""
SentinelEdge — Analytics Service
===============================
Aggregates sensor_readings and anomalies to compute uptime, state timelines,
durations, transition matrices, and confidence metrics for multi-device analytics.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from supabase_client import get_client

logger = logging.getLogger(__name__)

def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse ISO 8601 string to a timezone-aware datetime object."""
    try:
        # Standard format: 2026-06-30T10:00:00+00:00 or with Z
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    except Exception:
        return datetime.now(timezone.utc)

def get_analytics_summary_json(device_id: str, range_hours: int = 24) -> Dict[str, Any]:
    """
    Fetch and aggregate sensor_readings and anomalies for a specific device.
    Returns a structured analytics summary dictionary.
    """
    try:
        client = get_client()
        since = (datetime.now(timezone.utc) - timedelta(hours=range_hours)).isoformat()
        
        # 1. Fetch sensor_readings (baseline periodic state & environment)
        readings_res = client.table("sensor_readings") \
            .select("timestamp, motion_state, confidence, unknown_candidate, classification_margin, temperature, humidity") \
            .eq("device_id", device_id) \
            .gte("timestamp", since) \
            .order("timestamp", desc=False) \
            .execute()
        
        # 2. Fetch anomalies (precise fault transitions)
        anomalies_res = client.table("anomalies") \
            .select("timestamp, fault_label, confidence, classification_margin, unknown_candidate") \
            .eq("device_id", device_id) \
            .gte("timestamp", since) \
            .order("timestamp", desc=False) \
            .execute()
        
        readings = readings_res.data if readings_res.data else []
        anomalies = anomalies_res.data if anomalies_res.data else []
        
        logger.info(f"Loaded {len(readings)} readings and {len(anomalies)} anomalies for device {device_id}")

        # 3. Aggregate Confidence Statistics (from sensor_readings)
        confidences = [r["confidence"] for r in readings if r.get("confidence") is not None]
        margins = [r["classification_margin"] for r in readings if r.get("classification_margin") is not None]
        
        avg_confidence = round(sum(confidences) / len(confidences) * 100.0, 1) if confidences else None
        min_confidence = round(min(confidences) * 100.0, 1) if confidences else None
        avg_margin = round(sum(margins) / len(margins), 2) if margins else None

        # 4. Shadow Mode Unknown Candidate Statistics
        unknown_candidates = [r for r in readings if r.get("unknown_candidate") is True]
        total_candidates = len(unknown_candidates)
        
        last_candidate_ts = None
        if unknown_candidates:
            # Get latest candidate by timestamp
            sorted_candidates = sorted(unknown_candidates, key=lambda x: x["timestamp"])
            last_candidate_ts = sorted_candidates[-1]["timestamp"]
            
        candidate_rate_per_day = round((total_candidates / range_hours) * 24.0, 2) if range_hours > 0 else 0.0

        # 5. Build Combined Timeline
        # We merge both sources chronologically to construct transitions and durations.
        combined_events = []
        
        for r in readings:
            state = r.get("motion_state")
            if not state:
                state = "stationary" # default fallback
            combined_events.append({
                "timestamp": parse_iso_timestamp(r["timestamp"]),
                "state": state.lower()
            })
            
        for a in anomalies:
            state = a.get("fault_label")
            if not state:
                state = "unknown"
            combined_events.append({
                "timestamp": parse_iso_timestamp(a["timestamp"]),
                "state": state.lower()
            })

        # Sort chronologically
        combined_events.sort(key=lambda x: x["timestamp"])

        # Collapse sequential duplicate states and compute transitions/durations
        timeline = []
        transition_matrix = {}
        duration_seconds = {
            "stationary": 0.0,
            "movement": 0.0,
            "rotation": 0.0,
            "shake": 0.0
        }

        # Collapsed trace containing transitions
        collapsed_events = []
        for event in combined_events:
            if not collapsed_events or collapsed_events[-1]["state"] != event["state"]:
                collapsed_events.append(event)

        # Reconstruct transitions and calculate transition matrix
        for i in range(len(collapsed_events) - 1):
            s_from = collapsed_events[i]["state"]
            s_to = collapsed_events[i+1]["state"]
            # Filter out "unknown" from standard transition matrix
            if s_from != "unknown" and s_to != "unknown":
                trans_key = f"{s_from} → {s_to}"
                transition_matrix[trans_key] = transition_matrix.get(trans_key, 0) + 1

        # State Duration calculation (time between consecutive timestamps)
        for i in range(len(combined_events)):
            evt = combined_events[i]
            state = evt["state"]
            
            # Skip unknown from known state duration totals
            if state == "unknown" or state not in duration_seconds:
                continue
                
            # If not the last event, duration is delta to next event
            if i < len(combined_events) - 1:
                delta = (combined_events[i+1]["timestamp"] - evt["timestamp"]).total_seconds()
                # Cap extremely long durations (e.g. offline gaps > 15 mins) to prevent skewing
                if delta > 900.0:
                    delta = 5.0 # assume typical heartbeat interval
                duration_seconds[state] += delta
            else:
                # Last event: default to typical 5s heartbeat duration
                duration_seconds[state] += 5.0

        # Calculate percentages for known states
        total_known_seconds = sum(duration_seconds.values())
        known_duration_pct = {}
        for state, sec in duration_seconds.items():
            pct = round((sec / total_known_seconds) * 100.0, 1) if total_known_seconds > 0 else 0.0
            known_duration_pct[state] = pct

        # 6. Formulate State Timeline Output (chronological JSON list of transition points)
        timeline_output = []
        for evt in collapsed_events:
            timeline_output.append({
                "time": evt["timestamp"].strftime("%H:%M"),
                "state": evt["state"]
            })

        # Limit timeline output list size to prevent payload explosion
        timeline_output = timeline_output[-25:]

        # 7. Uptime calculation (percentage of expected heartbeats received)
        # Expected heartbeats in range_hours (heartbeat published every 5 seconds)
        expected_count = (range_hours * 3600) // 5
        uptime_pct = round((len(readings) / expected_count) * 100.0, 1) if expected_count > 0 else 0.0
        uptime_pct = min(uptime_pct, 100.0)

        # Environmental averages
        temps = [r["temperature"] for r in readings if r.get("temperature") is not None]
        hums = [r["humidity"] for r in readings if r.get("humidity") is not None]
        avg_temp = round(sum(temps) / len(temps), 1) if temps else None
        avg_hum = round(sum(hums) / len(hums), 1) if hums else None

        return {
            "device_id": device_id,
            "uptime_pct": uptime_pct,
            "known_duration_pct": known_duration_pct,
            "transition_matrix": transition_matrix,
            "shadow_unknown_stats": {
                "total_candidates": total_candidates,
                "last_candidate": last_candidate_ts,
                "candidate_rate_per_day": candidate_rate_per_day
            },
            "confidence_stats": {
                "avg_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "avg_classification_margin": avg_margin
            },
            "environment_stats": {
                "avg_temperature": avg_temp,
                "avg_humidity": avg_hum
            },
            "state_timeline": timeline_output,
            "total_anomalies": len(anomalies)
        }

    except Exception as e:
        logger.error(f"Error computing analytics summary for device {device_id}: {e}")
        return {
            "device_id": device_id,
            "uptime_pct": 0.0,
            "known_duration_pct": {"stationary": 0.0, "movement": 0.0, "rotation": 0.0, "shake": 0.0},
            "transition_matrix": {},
            "shadow_unknown_stats": {"total_candidates": 0, "last_candidate": None, "candidate_rate_per_day": 0.0},
            "confidence_stats": {"avg_confidence": None, "min_confidence": None, "avg_classification_margin": None},
            "environment_stats": {"avg_temperature": None, "avg_humidity": None},
            "state_timeline": [],
            "total_anomalies": 0
        }
