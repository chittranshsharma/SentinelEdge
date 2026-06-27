'use client'

import { type Anomaly, FAULT_LABELS, FAULT_COLORS } from '@/lib/supabase'

interface Props {
  anomalies: Anomaly[]
  loading:   boolean
}

function timeAgo(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime()
  const s = Math.floor(diff / 1000)
  if (s < 60)  return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  return `${Math.floor(s / 3600)}h ago`
}

export default function AnomalyFeed({ anomalies, loading }: Props) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5 h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300">Recent Anomalies</h2>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
          <span className="text-xs text-gray-500">Live</span>
        </div>
      </div>

      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-16 rounded-lg bg-gray-800/50 animate-pulse" />
          ))}
        </div>
      ) : anomalies.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-48 text-gray-600">
          <div className="text-3xl mb-2">✓</div>
          <p className="text-sm">No anomalies detected</p>
          <p className="text-xs mt-1">All systems nominal</p>
        </div>
      ) : (
        <div className="space-y-2">
          {anomalies.map((a) => {
            const color   = FAULT_COLORS[a.fault_class]
            const hasGps  = a.gps_lat !== null && a.gps_lng !== null
            return (
              <div
                key={a.id}
                className="flex items-start gap-3 p-3 rounded-lg bg-gray-800/40 hover:bg-gray-800/70 transition-colors"
                style={{ borderLeft: `2px solid ${color}` }}
              >
                <div
                  className="w-2 h-2 rounded-full flex-shrink-0 mt-1.5"
                  style={{ backgroundColor: color }}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <span className="text-sm font-medium text-white">
                      {FAULT_LABELS[a.fault_class]}
                    </span>
                    <span
                      className="text-xs font-mono px-1.5 py-0.5 rounded"
                      style={{ backgroundColor: `${color}20`, color }}
                    >
                      {(a.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5 flex flex-wrap gap-x-2">
                    <span>{timeAgo(a.created_at)}</span>
                    {hasGps && (
                      <span>{a.gps_lat!.toFixed(4)}°N {a.gps_lng!.toFixed(4)}°E</span>
                    )}
                  </div>
                  {a.llm_explanation && (
                    <p className="text-xs text-gray-400 mt-1.5 line-clamp-2">
                      {a.llm_explanation}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-2 mt-1 text-xs text-gray-600">
                    {a.accel_rms_x != null && <span>RMS {a.accel_rms_x.toFixed(2)}g</span>}
                    {a.temperature  != null && <span>· {a.temperature.toFixed(1)}°C</span>}
                    {a.air_quality_raw != null && <span>· AQ {a.air_quality_raw}</span>}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
