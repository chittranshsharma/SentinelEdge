-- Phase 3: Analytics and Shadow Unknown Detection Telemetry
-- Adds columns to track baseline states, classifier confidence, and shadow candidates.

ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS motion_state TEXT;
ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS confidence FLOAT;
ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS unknown_candidate BOOLEAN DEFAULT FALSE;
ALTER TABLE sensor_readings ADD COLUMN IF NOT EXISTS classification_margin FLOAT;
