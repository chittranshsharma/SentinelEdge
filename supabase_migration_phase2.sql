-- Phase 2: Environmental Telemetry Migration
-- Adds GPS satellite count and simulation mode flag.
-- NOTE: gas_percent is not stored (computed in UI).

ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS gps_satellites INTEGER;
ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS gps_simulated  BOOLEAN DEFAULT FALSE;
ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS dht_success INTEGER DEFAULT 0;
ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS dht_failure INTEGER DEFAULT 0;
