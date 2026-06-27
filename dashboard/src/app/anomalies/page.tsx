'use client'

import { useEffect, useState } from 'react'
import { supabase, type Anomaly, FAULT_LABELS, FAULT_COLORS } from '@/lib/supabase'

const FAULT_CLASSES = [0, 1, 2, 3]
const FAULT_FILTER_LABELS = ['All', ...FAULT_CLASSES.slice(1).map(c => FAULT_LABELS[c])]

function formatDateTime(ts: string) {
  return new Date(ts).toLocaleString('en-IN', {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

export default function AnomaliesPage() {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([])
  const [filter, setFilter]       = useState<number | null>(null)
  const [loading, setLoading]     = useState(true)
  const [page, setPage]           = useState(0)
  const PAGE_SIZE = 20

  const fetchAnomalies = async (faultClass: number | null, p: number) => {
    setLoading(true)
    let query = supabase
      .from('anomalies')
      .select('*')
      .order('created_at', { ascending: false })
      .range(p * PAGE_SIZE, (p + 1) * PAGE_SIZE - 1)

    if (faultClass !== null) {
      query = query.eq('fault_class', faultClass)
    }

    const { data } = await query
    setAnomalies(data ?? [])
    setLoading(false)
  }

  useEffect(() => { fetchAnomalies(filter, page) }, [filter, page])

  // Realtime new anomalies
  useEffect(() => {
    const channel = supabase
      .channel('anomalies-page')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'anomalies' },
        (payload) => {
          const a = payload.new as Anomaly
          if (filter === null || a.fault_class === filter) {
            setAnomalies((prev) => [a, ...prev].slice(0, PAGE_SIZE))
          }
        }
      )
      .subscribe()
    return () => { supabase.removeChannel(channel) }
  }, [filter])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Anomaly Feed</h1>
          <p className="text-sm text-gray-400 mt-0.5">All detected faults with confidence scores and LLM analysis</p>
        </div>
        <button
          onClick={() => fetchAnomalies(filter, page)}
          className="text-xs text-gray-400 hover:text-white border border-gray-700 hover:border-gray-500 rounded-lg px-3 py-1.5 transition-colors"
        >
          ↻ Refresh
        </button>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2">
        {[null, 1, 2, 3].map((cls, i) => (
          <button
            key={i}
            onClick={() => { setFilter(cls); setPage(0) }}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              filter === cls
                ? 'bg-violet-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            {cls === null ? 'All' : FAULT_LABELS[cls]}
          </button>
        ))}
      </div>

      {/* Anomaly list */}
      <div className="space-y-3">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-24 rounded-xl bg-gray-800/50 animate-pulse" />
          ))
        ) : anomalies.length === 0 ? (
          <div className="text-center py-16 text-gray-500">
            <div className="text-4xl mb-3">✓</div>
            <p>No anomalies detected</p>
          </div>
        ) : (
          anomalies.map((a) => (
            <AnomalyCard key={a.id} anomaly={a} />
          ))
        )}
      </div>

      {/* Pagination */}
      <div className="flex gap-3 justify-center pt-2">
        <button
          disabled={page === 0}
          onClick={() => setPage(p => p - 1)}
          className="px-4 py-2 text-sm bg-gray-800 text-gray-300 rounded-lg disabled:opacity-40 hover:bg-gray-700 transition-colors"
        >
          ← Previous
        </button>
        <span className="px-4 py-2 text-sm text-gray-400">Page {page + 1}</span>
        <button
          disabled={anomalies.length < PAGE_SIZE}
          onClick={() => setPage(p => p + 1)}
          className="px-4 py-2 text-sm bg-gray-800 text-gray-300 rounded-lg disabled:opacity-40 hover:bg-gray-700 transition-colors"
        >
          Next →
        </button>
      </div>
    </div>
  )
}

function AnomalyCard({ anomaly: a }: { anomaly: Anomaly }) {
  const color = FAULT_COLORS[a.fault_class] ?? '#6b7280'
  const hasGps = a.gps_lat !== null && a.gps_lng !== null

  return (
    <div
      className="rounded-xl border border-gray-800 bg-gray-900 p-4 hover:border-gray-700 transition-colors"
      style={{ borderLeft: `3px solid ${color}` }}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-2">
          <span
            className="w-2.5 h-2.5 rounded-full flex-shrink-0 mt-1"
            style={{ backgroundColor: color }}
          />
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold text-white">
                {FAULT_LABELS[a.fault_class]}
              </span>
              <span
                className="text-xs font-mono rounded px-1.5 py-0.5"
                style={{ backgroundColor: `${color}22`, color }}
              >
                {(a.confidence * 100).toFixed(0)}% confidence
              </span>
            </div>
            <div className="text-xs text-gray-500 mt-0.5 flex flex-wrap gap-2">
              <span>{formatDateTime(a.created_at)}</span>
              {hasGps && (
                <span>📍 {a.gps_lat!.toFixed(4)}°N, {a.gps_lng!.toFixed(4)}°E</span>
              )}
              {a.inference_latency_ms && (
                <span>⚡ {a.inference_latency_ms}ms</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* LLM Explanation */}
      {a.llm_explanation && (
        <div className="mt-3 text-sm text-gray-300 bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
          <span className="text-xs text-violet-400 font-medium">🤖 AI Analysis  </span>
          {a.llm_explanation}
        </div>
      )}

      {/* Sensor readings */}
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-500">
        {a.accel_rms_x != null && <span>Vibration: {a.accel_rms_x.toFixed(2)}g</span>}
        {a.temperature  != null && <span>Temp: {a.temperature.toFixed(1)}°C</span>}
        {a.humidity     != null && <span>Humidity: {a.humidity.toFixed(0)}%</span>}
        {a.air_quality_raw != null && <span>AQ: {a.air_quality_raw} ADC</span>}
      </div>
    </div>
  )
}
