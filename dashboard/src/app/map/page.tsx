'use client'

import dynamic from 'next/dynamic'

// Leaflet must be loaded client-side only (uses window/document)
const AnomalyMap = dynamic(() => import('@/components/AnomalyMap'), {
  ssr: false,
  loading: () => (
    <div className="h-[600px] rounded-xl bg-gray-800/50 animate-pulse flex items-center justify-center">
      <span className="text-gray-500">Loading map…</span>
    </div>
  ),
})

export default function MapPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold text-white">Anomaly Map</h1>
        <p className="text-sm text-gray-400 mt-0.5">
          GPS-tagged fault locations. Pin color indicates fault type.
        </p>
      </div>
      <AnomalyMap />
    </div>
  )
}
