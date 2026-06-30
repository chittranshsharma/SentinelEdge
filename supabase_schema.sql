-- SentinelEdge — Supabase Schema
-- ==================================
-- Run this in the Supabase SQL Editor (project → SQL Editor → New query)
-- After creating tables, enable Realtime on sensor_readings and anomalies.

-- ── sensor_readings ────────────────────────────────────────────────────────────
-- Every heartbeat + anomaly event writes here.
-- Used for: live dashboard readings, device online status, analytics.

CREATE TABLE IF NOT EXISTS sensor_readings (
  id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  device_id       TEXT        NOT NULL,
  timestamp       TIMESTAMPTZ NOT NULL,
  temperature     FLOAT,
  humidity        FLOAT,
  air_quality_raw INTEGER,
  accel_rms_x     FLOAT,
  accel_rms_y     FLOAT,
  accel_rms_z     FLOAT,
  gps_lat         FLOAT,
  gps_lng         FLOAT,
  gps_fix         BOOLEAN     DEFAULT FALSE,
  gps_satellites  INTEGER,
  gps_simulated   BOOLEAN     DEFAULT FALSE,
  dht_success     INTEGER     DEFAULT 0,
  dht_failure     INTEGER     DEFAULT 0,
  motion_state          TEXT,
  confidence            FLOAT,
  unknown_candidate     BOOLEAN     DEFAULT FALSE,
  classification_margin FLOAT,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast "last N readings" queries
CREATE INDEX IF NOT EXISTS idx_sensor_readings_device_time
  ON sensor_readings (device_id, created_at DESC);

-- ── anomalies ─────────────────────────────────────────────────────────────────
-- Every detected fault event (fault_class != 0, confidence > 0.75).
-- LLM explanation populated asynchronously after Groq API responds.

CREATE TABLE IF NOT EXISTS anomalies (
  id                    UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
  device_id             TEXT    NOT NULL,
  timestamp             TIMESTAMPTZ NOT NULL,
  fault_class           INTEGER NOT NULL,   -- 0=normal,1=imbalance,2=obstruction,3=loose_mount
  fault_label           TEXT    NOT NULL,
  confidence            FLOAT   NOT NULL,   -- 0.0..1.0
  inference_latency_ms  INTEGER,            -- on-device: feature_extraction + TFLite invoke
  accel_rms_x           FLOAT,
  accel_rms_y           FLOAT,
  accel_rms_z           FLOAT,
  temperature           FLOAT,
  humidity              FLOAT,
  air_quality_raw       INTEGER,
  gps_lat               FLOAT,
  gps_lng               FLOAT,
  llm_explanation       TEXT,               -- populated async by Groq
  telegram_sent         BOOLEAN DEFAULT FALSE,
  created_at            TIMESTAMPTZ DEFAULT NOW()
);

-- Index for map queries (get all anomalies with GPS coordinates)
CREATE INDEX IF NOT EXISTS idx_anomalies_gps
  ON anomalies (device_id, created_at DESC)
  WHERE gps_lat IS NOT NULL AND gps_lng IS NOT NULL;

-- Index for analytics time-series queries
CREATE INDEX IF NOT EXISTS idx_anomalies_time_class
  ON anomalies (fault_class, created_at DESC);

-- ── alerts ────────────────────────────────────────────────────────────────────
-- Log of all Telegram alerts sent (for audit trail).

CREATE TABLE IF NOT EXISTS alerts (
  id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  anomaly_id  UUID        REFERENCES anomalies(id) ON DELETE SET NULL,
  channel     TEXT        NOT NULL,   -- 'telegram'
  message     TEXT        NOT NULL,
  sent_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ── Row Level Security ────────────────────────────────────────────────────────
-- Enable RLS (backend uses service key which bypasses RLS)
-- Dashboard uses anon key — read-only access

ALTER TABLE sensor_readings ENABLE ROW LEVEL SECURITY;
ALTER TABLE anomalies       ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts          ENABLE ROW LEVEL SECURITY;

-- Anon can read all tables (dashboard)
-- DROP first so this block is safe to re-run (Postgres has no CREATE POLICY IF NOT EXISTS)
DROP POLICY IF EXISTS "Allow anon read on sensor_readings" ON sensor_readings;
CREATE POLICY "Allow anon read on sensor_readings"
  ON sensor_readings FOR SELECT TO anon USING (true);

DROP POLICY IF EXISTS "Allow anon read on anomalies" ON anomalies;
CREATE POLICY "Allow anon read on anomalies"
  ON anomalies FOR SELECT TO anon USING (true);

DROP POLICY IF EXISTS "Allow anon read on alerts" ON alerts;
CREATE POLICY "Allow anon read on alerts"
  ON alerts FOR SELECT TO anon USING (true);

-- ── Realtime ──────────────────────────────────────────────────────────────────
-- Enable realtime replication for live dashboard updates.
-- Wrapped in DO blocks so this is safe to re-run (no IF NOT EXISTS for publications).

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_publication_tables
    WHERE pubname = 'supabase_realtime' AND tablename = 'sensor_readings'
  ) THEN
    ALTER PUBLICATION supabase_realtime ADD TABLE sensor_readings;
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_publication_tables
    WHERE pubname = 'supabase_realtime' AND tablename = 'anomalies'
  ) THEN
    ALTER PUBLICATION supabase_realtime ADD TABLE anomalies;
  END IF;
END $$;
