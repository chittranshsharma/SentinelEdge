'use client'

import { useEffect, useState } from 'react'
import { type SensorReading } from '@/lib/supabase'

interface Props {
  lastReading: SensorReading | null
}

export default function DeviceStatus({ lastReading }: Props) {
  const [mounted, setMounted] = useState(false)
  const [online, setOnline] = useState(false)
  const [secondsAgo, setSecondsAgo] = useState<number | null>(null)

  useEffect(() => {
    setMounted(true)
    const updateStatus = () => {
      if (!lastReading) return
      const lastSeen = new Date(lastReading.created_at).getTime()
      const diff = Math.floor((Date.now() - lastSeen) / 1000)
      setSecondsAgo(diff >= 0 ? diff : 0)
      setOnline(diff >= 0 && diff < 30)
    }
    updateStatus()
    const timer = setInterval(updateStatus, 1000)
    return () => clearInterval(timer)
  }, [lastReading])

  const statusText = !lastReading
    ? 'No data received'
    : !mounted
    ? 'Checking status...'
    : online
    ? 'Online'
    : `Last seen ${secondsAgo}s ago`

  return (
    <div className={`rounded-xl border p-4 flex items-center justify-between ${
      online
        ? 'border-green-800/50 bg-green-950/30'
        : lastReading
        ? 'border-yellow-800/50 bg-yellow-950/30'
        : 'border-gray-800 bg-gray-900'
    }`}>
      <div className="flex items-center gap-3">
        <div className={`relative flex-shrink-0`}>
          <div
            className={`w-3 h-3 rounded-full ${
              online ? 'bg-green-400' : lastReading ? 'bg-yellow-400' : 'bg-gray-500'
            }`}
          />
          {online && (
            <div className="absolute inset-0 rounded-full bg-green-400 animate-ping opacity-50" />
          )}
        </div>
        <div>
          <div className="text-sm font-semibold text-white">
            {online ? '🟢 Device Online' : lastReading ? '🟡 Device Offline' : '⚪ No Device'}
          </div>
          <div className="text-xs text-gray-400">{statusText}</div>
        </div>
      </div>
      <div className="text-right text-xs text-gray-500 font-mono">
        <div>sentineledge-001</div>
        {lastReading?.gps_lat && lastReading.gps_lng && (
          <div>{lastReading.gps_lat.toFixed(4)}°N, {lastReading.gps_lng.toFixed(4)}°E</div>
        )}
      </div>
    </div>
  )
}
