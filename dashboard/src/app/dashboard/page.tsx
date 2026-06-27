'use client'

import { useEffect, useState } from 'react'
import { supabase, type Anomaly, type SensorReading, FAULT_LABELS, FAULT_COLORS } from '@/lib/supabase'
import DeviceStatus from '@/components/DeviceStatus'
import LiveReadings from '@/components/LiveReadings'
import AnomalyFeed from '@/components/AnomalyFeed'

export default function DashboardPage() {
  const [anomalies, setAnomalies]   = useState<Anomaly[]>([])
  const [lastReading, setLastReading] = useState<SensorReading | null>(null)
  const [loading, setLoading]         = useState(true)

  // ── Initial data fetch ─────────────────────────────────────────────────────
  useEffect(() => {
    const fetchInitial = async () => {
      const [anomalyRes, readingRes] = await Promise.all([
        supabase
          .from('anomalies')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(10),
        supabase
          .from('sensor_readings')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(1),
      ])

      if (anomalyRes.data) setAnomalies(anomalyRes.data)
      if (readingRes.data?.[0]) setLastReading(readingRes.data[0])
      setLoading(false)
    }
    fetchInitial()
  }, [])

  // ── Realtime: anomalies INSERT ─────────────────────────────────────────────
  useEffect(() => {
    const channel = supabase
      .channel('dashboard-anomalies')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'anomalies' },
        (payload) => {
          setAnomalies((prev) => [payload.new as Anomaly, ...prev].slice(0, 10))
        }
      )
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [])

  // ── Realtime: sensor_readings INSERT (heartbeat) ───────────────────────────
  useEffect(() => {
    const channel = supabase
      .channel('dashboard-readings')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'sensor_readings' },
        (payload) => {
          setLastReading(payload.new as SensorReading)
        }
      )
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [])

  // ── Metric cards ───────────────────────────────────────────────────────────
  const totalAnomalies = anomalies.length
  const lastAnomaly    = anomalies[0] ?? null
  const avgConfidence  = anomalies.length
    ? (anomalies.reduce((s, a) => s + a.confidence, 0) / anomalies.length * 100).toFixed(0)
    : '--'

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-sm text-gray-400 mt-0.5">
          Real-time edge AI fault detection — ESP32 on-device TFLite inference
        </p>
      </div>

      {/* Device status bar */}
      <DeviceStatus lastReading={lastReading} />

      {/* KPI cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          {
            label: 'Anomalies (recent)',
            value: loading ? '…' : String(totalAnomalies),
            sub: 'last 10 events',
            accent: 'from-red-500/20 to-red-500/5',
            dot: 'bg-red-500',
          },
          {
            label: 'Last Fault',
            value: lastAnomaly ? FAULT_LABELS[lastAnomaly.fault_class] : 'None',
            sub: lastAnomaly
              ? `${(lastAnomaly.confidence * 100).toFixed(0)}% confidence`
              : 'all clear',
            accent: lastAnomaly
              ? 'from-orange-500/20 to-orange-500/5'
              : 'from-green-500/20 to-green-500/5',
            dot: lastAnomaly ? 'bg-orange-500' : 'bg-green-500',
          },
          {
            label: 'Avg Confidence',
            value: avgConfidence === '--' ? '--' : `${avgConfidence}%`,
            sub: 'fault predictions',
            accent: 'from-violet-500/20 to-violet-500/5',
            dot: 'bg-violet-500',
          },
          {
            label: 'Inference Latency',
            value: lastAnomaly?.inference_latency_ms
              ? `${lastAnomaly.inference_latency_ms}ms`
              : '--',
            sub: 'on-device (ESP32)',
            accent: 'from-cyan-500/20 to-cyan-500/5',
            dot: 'bg-cyan-500',
          },
        ].map((card) => (
          <div
            key={card.label}
            className={`relative rounded-xl border border-gray-800 bg-gradient-to-br ${card.accent} p-4 overflow-hidden`}
          >
            <div className={`absolute top-3 right-3 w-2 h-2 rounded-full ${card.dot} animate-pulse`} />
            <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">{card.label}</p>
            <p className="text-2xl font-bold text-white mt-1">{card.value}</p>
            <p className="text-xs text-gray-500 mt-0.5">{card.sub}</p>
          </div>
        ))}
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live readings — 1 col */}
        <div className="lg:col-span-1">
          <LiveReadings reading={lastReading} />
        </div>

        {/* Anomaly feed — 2 cols */}
        <div className="lg:col-span-2">
          <AnomalyFeed anomalies={anomalies} loading={loading} />
        </div>
      </div>
    </div>
  )
}
