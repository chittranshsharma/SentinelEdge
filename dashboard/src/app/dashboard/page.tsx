'use client'

import { useEffect, useState } from 'react'
import { supabase, type Anomaly, type SensorReading, FAULT_LABELS, FAULT_COLORS } from '@/lib/supabase'
import DeviceStatus from '@/components/DeviceStatus'
import LiveReadings from '@/components/LiveReadings'
import AnomalyFeed from '@/components/AnomalyFeed'
import EnvTelemetry from '@/components/EnvTelemetry'

export default function DashboardPage() {
  const [anomalies, setAnomalies]   = useState<Anomaly[]>([])
  const [lastReading, setLastReading] = useState<SensorReading | null>(null)
  const [loading, setLoading]         = useState(true)
  
  // Client-side calculations to avoid hydration mismatch
  const [mounted, setMounted] = useState(false)
  const [deviceOnline, setDeviceOnline] = useState(false)
  const [lastSeenTime, setLastSeenTime] = useState('')

  // ── Initial data fetch ─────────────────────────────────────────────────────
  useEffect(() => {
    const fetchInitial = async () => {
      const [anomalyRes, readingRes] = await Promise.all([
        supabase
          .from('anomalies')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(20),
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

  // Update status state on client
  useEffect(() => {
    setMounted(true)
    if (!lastReading) return
    const update = () => {
      const isOnline = new Date().getTime() - new Date(lastReading.created_at).getTime() < 120000
      setDeviceOnline(isOnline)
      setLastSeenTime(new Date(lastReading.created_at).toLocaleTimeString())
    }
    update()
    const timer = setInterval(update, 5000)
    return () => clearInterval(timer)
  }, [lastReading])

  // ── Realtime: anomalies INSERT ─────────────────────────────────────────────
  useEffect(() => {
    const channel = supabase
      .channel('dashboard-anomalies')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'anomalies' },
        (payload) => {
          setAnomalies((prev) => [payload.new as Anomaly, ...prev].slice(0, 20))
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
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Device Status */}
        <div className="relative rounded-xl border border-gray-800 bg-gradient-to-br from-gray-800/40 to-gray-900/40 p-6 overflow-hidden">
          <div className={`absolute top-4 right-4 w-3 h-3 rounded-full animate-pulse ${
            mounted && lastReading && deviceOnline
              ? 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]'
              : 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]'
          }`} />
          <p className="text-sm text-gray-400 font-medium uppercase tracking-wider mb-2">Device Status</p>
          <p className="text-3xl font-bold text-white">
            {!mounted || !lastReading ? 'WAITING' : (
              deviceOnline ? 'ONLINE' : 'OFFLINE'
            )}
          </p>
          <p className="text-sm text-gray-500 mt-2">
            {!mounted || !lastReading ? 'No telemetry received' : `Last seen: ${lastSeenTime}`}
          </p>
        </div>

        {/* Card 2: Current State */}
        <div className={`relative rounded-xl border border-gray-800 bg-gradient-to-br p-6 overflow-hidden ${
          lastAnomaly ? 'from-blue-500/20 to-blue-900/10' : 'from-gray-800/40 to-gray-900/40'
        }`}>
          <p className="text-sm text-gray-400 font-medium uppercase tracking-wider mb-2">Current State</p>
          <p className="text-3xl font-bold text-white uppercase">
            {lastAnomaly ? FAULT_LABELS[lastAnomaly.fault_class] : 'UNKNOWN'}
          </p>
          <p className="text-sm text-gray-500 mt-2">
            {lastAnomaly ? `Since ${new Date(lastAnomaly.timestamp).toLocaleTimeString()}` : 'Waiting for transition'}
          </p>
        </div>

        {/* Card 3: Confidence */}
        <div className="relative rounded-xl border border-gray-800 bg-gradient-to-br from-gray-800/40 to-gray-900/40 p-6 overflow-hidden">
          <p className="text-sm text-gray-400 font-medium uppercase tracking-wider mb-2">Confidence</p>
          <p className="text-3xl font-bold text-white">
            {lastAnomaly ? `${(lastAnomaly.confidence * 100).toFixed(0)}%` : '--'}
          </p>
          <p className="text-sm text-gray-500 mt-2">
            TFLite on ESP32
          </p>
        </div>
      </div>

      {/* Environmental Telemetry (Phase 2) */}
      <EnvTelemetry reading={lastReading} />

      {/* Main content: Recent Events Table */}
      <div className="mt-8">
        <h2 className="text-xl font-semibold text-white mb-4">Last 20 Events</h2>
        <AnomalyFeed anomalies={anomalies} loading={loading} />
      </div>
    </div>
  )
}
