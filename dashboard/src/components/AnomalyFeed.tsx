'use client'

import { useEffect, useState } from 'react'
import { type Anomaly, FAULT_LABELS, FAULT_COLORS } from '@/lib/supabase'

interface Props {
  anomalies: Anomaly[]
  loading:   boolean
}

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`
  return `${Math.floor(s / 3600)}h`
}

export default function AnomalyFeed({ anomalies, loading }: Props) {
  const [now, setNow] = useState(Date.now())

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-gray-400">
          <thead className="bg-gray-800/50 text-xs uppercase font-medium text-gray-500">
            <tr>
              <th className="px-6 py-4">State</th>
              <th className="px-6 py-4">Confidence</th>
              <th className="px-6 py-4">Timestamp</th>
              <th className="px-6 py-4">Duration</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i} className="animate-pulse">
                  <td className="px-6 py-4"><div className="h-4 bg-gray-800 rounded w-24"></div></td>
                  <td className="px-6 py-4"><div className="h-4 bg-gray-800 rounded w-12"></div></td>
                  <td className="px-6 py-4"><div className="h-4 bg-gray-800 rounded w-32"></div></td>
                  <td className="px-6 py-4"><div className="h-4 bg-gray-800 rounded w-16"></div></td>
                </tr>
              ))
            ) : anomalies.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-6 py-8 text-center text-gray-500">
                  No events recorded
                </td>
              </tr>
            ) : (
              anomalies.map((a, i) => {
                const color = FAULT_COLORS[a.fault_class] || '#ffffff'
                const t = new Date(a.timestamp).getTime()
                
                // Duration: diff to next event, or diff to NOW for the current (index 0) event
                const durationMs = i === 0 
                  ? now - t
                  : new Date(anomalies[i - 1].timestamp).getTime() - t

                return (
                  <tr key={a.id} className="hover:bg-gray-800/20 transition-colors">
                    <td className="px-6 py-4 font-medium text-white flex items-center gap-2 uppercase">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                      {FAULT_LABELS[a.fault_class] || 'UNKNOWN'}
                    </td>
                    <td className="px-6 py-4 font-mono">
                      <span className="px-2 py-1 rounded" style={{ backgroundColor: `${color}20`, color }}>
                        {(a.confidence * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {new Date(a.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-6 py-4 font-mono text-gray-300">
                      {formatDuration(durationMs)}
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
