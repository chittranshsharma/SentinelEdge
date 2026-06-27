'use client'

import { type SensorReading } from '@/lib/supabase'

interface Props {
  reading: SensorReading | null
}

interface Metric {
  label: string
  value: string
  unit:  string
  icon:  string
  color: string
}

export default function LiveReadings({ reading }: Props) {
  const metrics: Metric[] = [
    {
      label: 'Temperature',
      value: reading?.temperature != null ? reading.temperature.toFixed(1) : '--',
      unit:  '°C',
      icon:  '🌡',
      color: 'text-orange-400',
    },
    {
      label: 'Humidity',
      value: reading?.humidity != null ? reading.humidity.toFixed(0) : '--',
      unit:  '%',
      icon:  '💧',
      color: 'text-blue-400',
    },
    {
      label: 'Air Quality',
      value: reading?.air_quality_raw != null ? String(reading.air_quality_raw) : '--',
      unit:  'ADC',
      icon:  '🌬',
      color: 'text-teal-400',
    },
    {
      label: 'Vibration RMS X',
      value: reading?.accel_rms_x != null ? reading.accel_rms_x.toFixed(3) : '--',
      unit:  'g',
      icon:  '📳',
      color: 'text-violet-400',
    },
    {
      label: 'Vibration RMS Y',
      value: reading?.accel_rms_y != null ? reading.accel_rms_y.toFixed(3) : '--',
      unit:  'g',
      icon:  '📳',
      color: 'text-violet-300',
    },
    {
      label: 'Vibration RMS Z',
      value: reading?.accel_rms_z != null ? reading.accel_rms_z.toFixed(3) : '--',
      unit:  'g',
      icon:  '📳',
      color: 'text-violet-200',
    },
    {
      label: 'GPS',
      value: reading?.gps_fix
        ? `${reading.gps_lat?.toFixed(4)}° ${reading.gps_lng?.toFixed(4)}°`
        : 'No fix',
      unit:  '',
      icon:  '📍',
      color: 'text-cyan-400',
    },
  ]

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5 h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300">Live Sensor Readings</h2>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs text-gray-500">Realtime</span>
        </div>
      </div>

      <div className="space-y-3">
        {metrics.map((m) => (
          <div key={m.label} className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span>{m.icon}</span>
              <span>{m.label}</span>
            </div>
            <div className={`font-mono text-sm font-semibold ${m.color}`}>
              {m.value}
              {m.unit && <span className="text-xs text-gray-500 ml-1">{m.unit}</span>}
            </div>
          </div>
        ))}
      </div>

      {reading && (
        <div className="mt-4 pt-3 border-t border-gray-800 text-xs text-gray-600">
          Last updated: {new Date(reading.created_at).toLocaleTimeString()}
        </div>
      )}
    </div>
  )
}
