import { createClient } from '@supabase/supabase-js'

const supabaseUrl  = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://placeholder.supabase.co'
const supabaseKey  = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'placeholder'

// Singleton client — shared across all components
export const supabase = createClient(supabaseUrl, supabaseKey, {
  realtime: {
    params: {
      eventsPerSecond: 10,
    },
  },
})

// ── Type Definitions ───────────────────────────────────────────────────────────

export interface SensorReading {
  id:              string
  device_id:       string
  timestamp:       string
  temperature:     number | null
  humidity:        number | null
  air_quality_raw: number | null
  accel_rms_x:     number | null
  accel_rms_y:     number | null
  accel_rms_z:     number | null
  gps_lat:         number | null
  gps_lng:         number | null
  gps_fix:         boolean
  gps_satellites:  number | null
  gps_simulated:   boolean | null
  dht_success:     number | null
  dht_failure:     number | null
  created_at:      string
}

// Helper: convert raw 12-bit ADC (0-4095) to 0-100%
export function gasToPercent(raw: number | null): number | null {
  if (raw === null || raw === undefined) return null
  return Math.round((raw / 4095) * 100)
}

export interface Anomaly {
  id:                   string
  device_id:            string
  timestamp:            string
  fault_class:          number
  fault_label:          string
  confidence:           number
  inference_latency_ms: number | null
  accel_rms_x:          number | null
  accel_rms_y:          number | null
  accel_rms_z:          number | null
  temperature:          number | null
  humidity:             number | null
  air_quality_raw:      number | null
  gps_lat:              number | null
  gps_lng:              number | null
  llm_explanation:      string | null
  telegram_sent:        boolean
  created_at:           string
}

// ── Constants ──────────────────────────────────────────────────────────────────

export const FAULT_LABELS: Record<number, string> = {
  0: 'Stationary',
  1: 'Movement',
  2: 'Rotation',
  3: 'Shake',
}

export const FAULT_COLORS: Record<number, string> = {
  0: '#22c55e',   // green
  1: '#ef4444',   // red
  2: '#f97316',   // orange
  3: '#a855f7',   // purple
}
