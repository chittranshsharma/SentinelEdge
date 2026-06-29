"""
SentinelEdge — Pydantic Data Models
=====================================
Type-safe models for MQTT message parsing and Supabase record schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SensorPayload(BaseModel):
    """Sensors block nested inside anomaly and heartbeat MQTT payloads."""
    accel_rms_x:    Optional[float] = None
    accel_rms_y:    Optional[float] = None
    accel_rms_z:    Optional[float] = None
    temperature:    Optional[float] = None
    humidity:       Optional[float] = None
    air_quality_raw: Optional[int]  = None
    gps_lat:        Optional[float] = None
    gps_lng:        Optional[float] = None
    gps_fix:        Optional[bool]  = False
    gps_satellites: Optional[int]   = None
    gps_simulated:  Optional[bool]  = False


class AnomalyPayload(BaseModel):
    """
    MQTT payload for sentineledge/anomaly topic.
    Matches the JSON schema in firmware/src/mqtt_handler.cpp.
    """
    device_id:            str
    timestamp:            int   # Unix epoch seconds (or millis/1000 fallback)
    fault_class:          int   # 0=normal, 1=imbalance, 2=obstruction, 3=loose_mount
    fault_label:          str
    confidence:           float # [0.0, 1.0]
    inference_latency_ms: Optional[int]  = None
    sensors:              SensorPayload  = Field(default_factory=SensorPayload)


class HeartbeatPayload(BaseModel):
    """MQTT payload for sentineledge/heartbeat topic."""
    device_id:   str
    timestamp:   int
    uptime_ms:   Optional[int]  = None
    free_heap:   Optional[int]  = None
    sensors:     SensorPayload  = Field(default_factory=SensorPayload)


class SensorReadingRecord(BaseModel):
    """Record to insert into Supabase sensor_readings table."""
    device_id:       str
    timestamp:       datetime
    temperature:     Optional[float] = None
    humidity:        Optional[float] = None
    air_quality_raw: Optional[int]   = None
    accel_rms_x:     Optional[float] = None
    accel_rms_y:     Optional[float] = None
    accel_rms_z:     Optional[float] = None
    gps_lat:         Optional[float] = None
    gps_lng:         Optional[float] = None
    gps_fix:         Optional[bool]  = False


class AnomalyRecord(BaseModel):
    """Record to insert into Supabase anomalies table."""
    device_id:            str
    timestamp:            datetime
    fault_class:          int
    fault_label:          str
    confidence:           float
    inference_latency_ms: Optional[int]   = None
    accel_rms_x:          Optional[float] = None
    accel_rms_y:          Optional[float] = None
    accel_rms_z:          Optional[float] = None
    temperature:          Optional[float] = None
    humidity:             Optional[float] = None
    air_quality_raw:      Optional[int]   = None
    gps_lat:              Optional[float] = None
    gps_lng:              Optional[float] = None
    llm_explanation:      Optional[str]   = None
    telegram_sent:        bool = False


FAULT_LABELS = {
    0: "stationary",
    1: "movement",
    2: "rotation",
    3: "shake",
}
