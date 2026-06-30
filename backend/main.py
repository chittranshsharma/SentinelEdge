"""
SentinelEdge — FastAPI Backend
================================
Main application entry point.

Routes:
  GET /                → health check
  GET /api/anomalies   → last N anomalies (paginated)
  GET /api/readings    → last N sensor readings
  GET /api/status      → device online/offline state

MQTT client starts in background on lifespan startup.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
import mqtt_client as mqtt
from supabase_client import get_client
import analytics_routes

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── MQTT Client (module-level reference for cleanup) ───────────────────────────
_mqtt_client = None


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start MQTT client on app startup, stop on shutdown."""
    global _mqtt_client
    logger.info("SentinelEdge backend starting...")

    # Verify Supabase connection
    try:
        client = get_client()
        client.table("sensor_readings").select("id").limit(1).execute()
        logger.info("Supabase connection verified")
    except Exception as e:
        logger.warning(f"Supabase check failed (will retry): {e}")

    # Start MQTT client with the current event loop
    loop = asyncio.get_running_loop()
    _mqtt_client = mqtt.start_mqtt(loop)
    logger.info("MQTT client started")

    yield  # Application runs here

    # Cleanup
    if _mqtt_client:
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()
        logger.info("MQTT client stopped")
    logger.info("SentinelEdge backend stopped")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SentinelEdge API",
    description="Edge AI fault detection backend — ESP32 TinyML → Raspberry Pi → Supabase",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten for production
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(analytics_routes.router)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service":   "SentinelEdge Backend",
        "status":    "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mqtt":      "connected" if _mqtt_client and _mqtt_client.is_connected() else "disconnected",
    }


@app.get("/api/anomalies", tags=["Data"])
async def get_anomalies(limit: int = 50, offset: int = 0):
    """Return recent anomalies, newest first."""
    try:
        client = get_client()
        result = (
            client.table("anomalies")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return {"anomalies": result.data, "count": len(result.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/readings", tags=["Data"])
async def get_readings(limit: int = 100, offset: int = 0):
    """Return recent sensor readings, newest first."""
    try:
        client = get_client()
        result = (
            client.table("sensor_readings")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return {"readings": result.data, "count": len(result.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status", tags=["Device"])
async def get_device_status():
    """
    Return device online/offline status.
    Device is considered online if a heartbeat was received within 30 seconds.
    """
    try:
        client = get_client()
        result = (
            client.table("sensor_readings")
            .select("created_at, device_id")
            .eq("device_id", settings.device_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if not result.data:
            return {"online": False, "last_seen": None}

        last_seen_str = result.data[0]["created_at"]
        last_seen     = datetime.fromisoformat(last_seen_str.replace("Z", "+00:00"))
        now           = datetime.now(timezone.utc)
        seconds_ago   = (now - last_seen).total_seconds()

        return {
            "online":     seconds_ago < 30,
            "last_seen":  last_seen_str,
            "seconds_ago": int(seconds_ago),
            "device_id":  settings.device_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
