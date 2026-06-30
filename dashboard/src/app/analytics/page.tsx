'use client'

import { useEffect, useState } from 'react'
import AnalyticsIntelligence from '@/components/AnalyticsIntelligence'

interface AnalyticsMetrics {
  device_id: string
  uptime_pct: number
  known_duration_pct: {
    stationary: number
    movement: number
    rotation: number
    shake: number
  }
  transition_matrix: Record<string, number>
  shadow_unknown_stats: {
    total_candidates: number
    last_candidate: string | null
    candidate_rate_per_day: number
  }
  confidence_stats: {
    avg_confidence: number | null
    min_confidence: number | null
    avg_classification_margin: number | null
  }
  environment_stats: {
    avg_temperature: number | null
    avg_humidity: number | null
  }
  state_timeline: Array<{
    time: string
    state: string
  }>
  total_anomalies: number
}

const FAULT_COLORS: Record<string, string> = {
  stationary: 'bg-green-500 text-green-500',
  movement: 'bg-blue-400 text-blue-400',
  rotation: 'bg-yellow-500 text-yellow-500',
  shake: 'bg-red-500 text-red-500',
  unknown: 'bg-violet-500 text-violet-500'
}

export default function AnalyticsPage() {
  const [deviceId, setDeviceId] = useState('sentineledge-001')
  const [metrics, setMetrics] = useState<AnalyticsMetrics | null>(null)
  const [summary, setSummary] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true)
      setError(null)
      try {
        const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
        const res = await fetch(`${apiBase}/api/analytics/${deviceId}/summary`)
        if (!res.ok) throw new Error('Failed to load analytics summary from backend')
        const data = await res.json()
        setMetrics(data.metrics)
        setSummary(data.summary)
      } catch (err: any) {
        console.error(err)
        setError(err.message || 'Error connecting to analytics server.')
      } finally {
        setLoading(false)
      }
    }
    fetchAnalytics()
  }, [deviceId, refreshTrigger])

  const handleRefresh = () => {
    setRefreshTrigger(prev => prev + 1)
  }

  if (loading && !metrics) {
    return (
      <div className="flex h-[400px] items-center justify-center">
        <div className="text-center space-y-3">
          <div className="w-10 h-10 border-4 border-violet-600 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-gray-400 text-sm">Aggregating historical metrics & compiling AI summary...</p>
        </div>
      </div>
    )
  }

  if (error && !metrics) {
    return (
      <div className="rounded-xl border border-red-900 bg-red-950/20 p-6 text-center max-w-lg mx-auto mt-12">
        <h2 className="text-red-400 font-semibold text-lg mb-2">Connection Error</h2>
        <p className="text-gray-300 text-sm mb-4">{error}</p>
        <button 
          onClick={handleRefresh}
          className="rounded-lg bg-red-800 hover:bg-red-700 px-4 py-2 text-sm text-white transition-colors"
        >
          Retry Connection
        </button>
      </div>
    )
  }

  const m = metrics!

  return (
    <div className="space-y-6">
      {/* 🏷️ Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <span className="text-violet-500">📊</span> Operational Analytics
          </h1>
          <p className="text-sm text-gray-400 mt-0.5">Device intelligence, transition summaries, and shadow-mode status</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Device ID:</span>
          <select 
            value={deviceId} 
            onChange={(e) => setDeviceId(e.target.value)}
            className="rounded-lg border border-gray-800 bg-gray-900 px-3 py-1.5 text-sm text-white focus:outline-none focus:border-violet-600"
          >
            <option value="sentineledge-001">sentineledge-001 (Fan node)</option>
          </select>
          <button 
            onClick={handleRefresh}
            className="p-2 text-gray-400 hover:text-white border border-gray-800 hover:border-gray-700 bg-gray-900 rounded-lg transition-colors"
            title="Refresh metrics data"
          >
            🔄
          </button>
        </div>
      </div>

      {/* 📈 Key Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-400 mb-1">System Uptime</p>
          <p className="text-2xl font-bold text-green-400">{m.uptime_pct}%</p>
          <p className="text-[10px] text-gray-500 mt-0.5">Last 24 hours</p>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-400 mb-1">Average Classifier Confidence</p>
          <p className="text-2xl font-bold text-violet-400">
            {m.confidence_stats.avg_confidence !== null ? `${m.confidence_stats.avg_confidence}%` : '--'}
          </p>
          <p className="text-[10px] text-gray-500 mt-0.5">
            Min: {m.confidence_stats.min_confidence !== null ? `${m.confidence_stats.min_confidence}%` : '--'}
          </p>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-400 mb-1">Vibration Margin (Top1-Top2)</p>
          <p className="text-2xl font-bold text-cyan-400">
            {m.confidence_stats.avg_classification_margin !== null ? m.confidence_stats.avg_classification_margin : '--'}
          </p>
          <p className="text-[10px] text-gray-500 mt-0.5">Mean classification delta</p>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-400 mb-1">Total Anomalies</p>
          <p className="text-2xl font-bold text-red-400">{m.total_anomalies}</p>
          <p className="text-[10px] text-gray-500 mt-0.5">Logged triggers</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 🕒 Known Activity Durations */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Known Activity Durations</h3>
            <div className="space-y-4">
              {Object.entries(m.known_duration_pct).map(([state, pct]) => {
                const colorClass = FAULT_COLORS[state] || 'bg-gray-600 text-gray-600'
                return (
                  <div key={state} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="capitalize font-medium text-gray-300">{state}</span>
                      <span className="text-gray-400">{pct}%</span>
                    </div>
                    <div className="w-full bg-gray-950 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${colorClass.split(' ')[0]} transition-all duration-500`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
          <div className="mt-6 text-[10px] text-gray-500 leading-normal">
            Calculated as relative share of known operation states from database telemetry history.
          </div>
        </div>

        {/* ⚠️ Shadow Mode Unknown Behavior Panel */}
        <div className="rounded-xl border border-gray-800 bg-gradient-to-br from-gray-900/50 to-violet-950/10 p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-violet-500 animate-pulse" /> Shadow Mode Unknown Detection
            </h3>
            
            <div className="space-y-4 mt-2">
              <div className="flex justify-between py-2 border-b border-gray-800/60">
                <span className="text-xs text-gray-400">Flagged Unknown Candidates</span>
                <span className="text-sm font-bold text-violet-400">{m.shadow_unknown_stats.total_candidates}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-800/60">
                <span className="text-xs text-gray-400">Trigger Rate / Day</span>
                <span className="text-sm font-bold text-white">{m.shadow_unknown_stats.candidate_rate_per_day} events</span>
              </div>
              <div className="py-1">
                <span className="text-xs text-gray-400 block mb-1">Last Candidate Logged:</span>
                <span className="text-xs font-mono text-gray-300 block">
                  {m.shadow_unknown_stats.last_candidate 
                    ? new Date(m.shadow_unknown_stats.last_candidate).toLocaleString()
                    : 'None recorded'}
                </span>
              </div>
            </div>
          </div>

          <div className="mt-6 border border-violet-850 bg-violet-950/20 rounded-lg p-3 text-[10px] text-violet-300 leading-relaxed">
            <strong>Shadow Rule:</strong> Flags candidates when confidence falls under 60% or when the gap between the top two classifications is narrower than 15%. Normal outputs continue to publish.
          </div>
        </div>

        {/* 📋 Chronological State Timeline */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">State Timeline</h3>
            <div className="space-y-3 max-h-[220px] overflow-y-auto pr-1 scrollbar-thin">
              {m.state_timeline.length === 0 ? (
                <p className="text-xs text-gray-500 italic text-center mt-12">No state changes logged in last 24h.</p>
              ) : (
                m.state_timeline.map((item, idx) => {
                  const colorClass = FAULT_COLORS[item.state] || 'text-gray-500'
                  return (
                    <div key={idx} className="flex items-center gap-3 text-xs py-1 border-b border-gray-800/40 last:border-0">
                      <span className="font-mono text-gray-500">{item.time}</span>
                      <span className="text-gray-400">→</span>
                      <span className={`font-semibold capitalize ${colorClass.split(' ').slice(1).join(' ')}`}>
                        {item.state}
                      </span>
                    </div>
                  )
                })
              )}
            </div>
          </div>
          <div className="mt-4 text-[10px] text-gray-500 leading-normal">
            Lists chronological sequence transitions, filtering out adjacent duplicates.
          </div>
        </div>
      </div>

      {/* 🧮 Transition Matrix */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-6">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Known Transition Matrix</h3>
        {Object.keys(m.transition_matrix).length === 0 ? (
          <p className="text-xs text-gray-500 italic py-6 text-center">No known transitions logged in the active window.</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {Object.entries(m.transition_matrix).map(([trans, count]) => (
              <div key={trans} className="rounded-lg border border-gray-800 bg-black/30 p-3 text-center">
                <p className="text-[10px] font-medium text-gray-400 capitalize mb-1">{trans}</p>
                <p className="text-xl font-bold text-white">{count}</p>
                <p className="text-[9px] text-gray-600">transitions</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 🤖 Groq Analytical Summaries and chat */}
      <AnalyticsIntelligence 
        deviceId={deviceId} 
        summary={summary} 
        onRefresh={handleRefresh}
      />
    </div>
  )
}
