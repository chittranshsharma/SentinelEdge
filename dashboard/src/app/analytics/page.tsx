'use client'

import { useEffect, useState } from 'react'
import { supabase, type Anomaly, FAULT_LABELS, FAULT_COLORS } from '@/lib/supabase'
import AnalyticsChart from '@/components/AnalyticsChart'

interface DailyCount {
  date: string
  normal: number
  imbalance: number
  obstruction: number
  loose_mount: number
}

export default function AnalyticsPage() {
  const [anomalies, setAnomalies]  = useState<Anomaly[]>([])
  const [loading, setLoading]      = useState(true)
  const [timeRange, setTimeRange]  = useState<'24h' | '7d' | '30d'>('7d')

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)

      const hours = timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 720
      const since = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString()

      const { data } = await supabase
        .from('anomalies')
        .select('*')
        .gte('created_at', since)
        .order('created_at', { ascending: true })
        .limit(500)

      setAnomalies(data ?? [])
      setLoading(false)
    }
    fetchData()
  }, [timeRange])

  // Aggregate into daily/hourly buckets
  const chartData = buildChartData(anomalies, timeRange)

  // Summary stats
  const totalByClass = [0, 1, 2, 3].map((cls) => ({
    label: FAULT_LABELS[cls],
    count: anomalies.filter(a => a.fault_class === cls).length,
    color: FAULT_COLORS[cls],
  }))

  const avgLatency = anomalies.length
    ? (anomalies
        .filter(a => a.inference_latency_ms)
        .reduce((s, a) => s + (a.inference_latency_ms ?? 0), 0)
      / anomalies.filter(a => a.inference_latency_ms).length).toFixed(0)
    : '--'

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Analytics</h1>
          <p className="text-sm text-gray-400 mt-0.5">Anomaly trends and performance metrics</p>
        </div>
        <div className="flex gap-2">
          {(['24h', '7d', '30d'] as const).map((r) => (
            <button
              key={r}
              onClick={() => setTimeRange(r)}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                timeRange === r
                  ? 'bg-violet-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Class breakdown */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {totalByClass.map(({ label, count, color }) => (
          <div key={label} className="rounded-xl border border-gray-800 bg-gray-900 p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-xs text-gray-400">{label}</span>
            </div>
            <p className="text-3xl font-bold text-white">{loading ? '…' : count}</p>
            <p className="text-xs text-gray-500 mt-0.5">events in {timeRange}</p>
          </div>
        ))}
      </div>

      {/* Time-series chart */}
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">
          Anomalies Over Time — {timeRange}
        </h2>
        {loading ? (
          <div className="h-64 bg-gray-800/50 rounded-lg animate-pulse" />
        ) : (
          <AnalyticsChart data={chartData} />
        )}
      </div>

      {/* Performance metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-400 mb-1">Avg Inference Latency</p>
          <p className="text-2xl font-bold text-cyan-400">
            {loading ? '…' : avgLatency === '--' ? '--' : `${avgLatency}ms`}
          </p>
          <p className="text-xs text-gray-500">on-device (ESP32)</p>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-400 mb-1">Total Events</p>
          <p className="text-2xl font-bold text-white">{loading ? '…' : anomalies.length}</p>
          <p className="text-xs text-gray-500">in {timeRange} window</p>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-400 mb-1">Avg Confidence</p>
          <p className="text-2xl font-bold text-violet-400">
            {loading || !anomalies.length
              ? '…'
              : `${(anomalies.reduce((s, a) => s + a.confidence, 0) / anomalies.length * 100).toFixed(0)}%`}
          </p>
          <p className="text-xs text-gray-500">fault classification</p>
        </div>
      </div>
    </div>
  )
}

function buildChartData(anomalies: Anomaly[], range: string): DailyCount[] {
  if (!anomalies.length) return []

  const bucketMs = range === '24h' ? 3600_000 : 86400_000  // 1h or 1d buckets
  const buckets  = new Map<string, DailyCount>()

  for (const a of anomalies) {
    const t   = new Date(a.created_at)
    const key = range === '24h'
      ? `${t.toLocaleDateString()} ${t.getHours()}:00`
      : t.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })

    if (!buckets.has(key)) {
      buckets.set(key, { date: key, normal: 0, imbalance: 0, obstruction: 0, loose_mount: 0 })
    }
    const b = buckets.get(key)!
    const k = ['normal', 'imbalance', 'obstruction', 'loose_mount'][a.fault_class] as keyof DailyCount
    ;(b as any)[k]++
  }

  return Array.from(buckets.values())
}
