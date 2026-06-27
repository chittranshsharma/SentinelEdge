'use client'

import { useEffect, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { supabase, type Anomaly, FAULT_LABELS, FAULT_COLORS } from '@/lib/supabase'

// Re-center map when anomalies load
function MapController({ center }: { center: [number, number] }) {
  const map = useMap()
  useEffect(() => { map.setView(center, map.getZoom()) }, [center, map])
  return null
}

export default function AnomalyMap() {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([])
  const [loading, setLoading]     = useState(true)

  useEffect(() => {
    const fetch = async () => {
      const { data } = await supabase
        .from('anomalies')
        .select('*')
        .not('gps_lat', 'is', null)
        .not('gps_lng', 'is', null)
        .order('created_at', { ascending: false })
        .limit(200)

      setAnomalies(data ?? [])
      setLoading(false)
    }
    fetch()
  }, [])

  // Realtime new anomalies with GPS
  useEffect(() => {
    const channel = supabase
      .channel('map-anomalies')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'anomalies' },
        (payload) => {
          const a = payload.new as Anomaly
          if (a.gps_lat && a.gps_lng) {
            setAnomalies((prev) => [a, ...prev].slice(0, 200))
          }
        }
      )
      .subscribe()
    return () => { supabase.removeChannel(channel) }
  }, [])

  // Center on last known GPS or default (New Delhi)
  const lastGps = anomalies.find(a => a.gps_lat && a.gps_lng)
  const center: [number, number] = lastGps
    ? [lastGps.gps_lat!, lastGps.gps_lng!]
    : [28.6139, 77.2090]

  return (
    <div className="space-y-3">
      {/* Legend */}
      <div className="flex flex-wrap gap-4">
        {[1, 2, 3].map((cls) => (
          <div key={cls} className="flex items-center gap-1.5 text-xs text-gray-400">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: FAULT_COLORS[cls] }}
            />
            {FAULT_LABELS[cls]}
          </div>
        ))}
        <span className="text-xs text-gray-600 ml-auto">
          {loading ? '…' : `${anomalies.length} pins`}
        </span>
      </div>

      {/* Leaflet Map */}
      <div className="rounded-xl overflow-hidden border border-gray-800" style={{ height: '580px' }}>
        <MapContainer
          center={center}
          zoom={13}
          style={{ height: '100%', width: '100%' }}
          className="z-0"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          <MapController center={center} />

          {anomalies.map((a) => (
            <CircleMarker
              key={a.id}
              center={[a.gps_lat!, a.gps_lng!]}
              radius={8}
              pathOptions={{
                color:       FAULT_COLORS[a.fault_class],
                fillColor:   FAULT_COLORS[a.fault_class],
                fillOpacity: 0.7,
                weight:      2,
              }}
            >
              <Popup maxWidth={280}>
                <div className="text-sm space-y-1.5">
                  <div className="font-bold" style={{ color: FAULT_COLORS[a.fault_class] }}>
                    {FAULT_LABELS[a.fault_class]}
                  </div>
                  <div className="text-gray-600">
                    {(a.confidence * 100).toFixed(0)}% confidence
                  </div>
                  <div className="text-gray-600 text-xs">
                    {new Date(a.created_at).toLocaleString()}
                  </div>
                  {a.inference_latency_ms && (
                    <div className="text-xs text-gray-600">
                      ⚡ {a.inference_latency_ms}ms on-device
                    </div>
                  )}
                  {a.llm_explanation && (
                    <div className="text-xs text-gray-700 mt-2 pt-2 border-t border-gray-200">
                      {a.llm_explanation}
                    </div>
                  )}
                  <div className="text-xs text-gray-500 font-mono">
                    {a.gps_lat!.toFixed(6)}, {a.gps_lng!.toFixed(6)}
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  )
}
