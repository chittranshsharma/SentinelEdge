import { SensorReading, gasToPercent } from '@/lib/supabase'

export default function EnvTelemetry({ reading }: { reading: SensorReading | null }) {
  // Graceful degradation when no data is available yet
  if (!reading) {
    return (
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="rounded-xl border border-gray-800 bg-gray-900/50 p-6 animate-pulse h-32" />
        ))}
      </div>
    )
  }

  // Derived values
  const temp = reading.temperature
  const hum = reading.humidity
  const gasRaw = reading.air_quality_raw
  const gasPct = gasToPercent(gasRaw)

  // Color logic
  const tempColor = temp === null ? 'text-gray-500' : temp < 27 ? 'text-green-500' : temp < 35 ? 'text-yellow-500' : 'text-red-500'
  const gasColor = gasPct === null ? 'bg-gray-800' : gasPct < 40 ? 'bg-green-500' : gasPct < 70 ? 'bg-yellow-500' : 'bg-red-500'

  return (
    <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      
      {/* 🌡️ Temperature */}
      <div className="rounded-xl border border-gray-800 bg-gradient-to-br from-gray-800/40 to-gray-900/40 p-6 flex flex-col justify-between">
        <p className="text-sm text-gray-400 font-medium uppercase tracking-wider">Temperature</p>
        <div className="mt-2 flex items-baseline gap-2">
          <p className={`text-3xl font-bold ${tempColor}`}>
            {temp !== null ? `${temp.toFixed(1)}°` : '--'}
          </p>
          <span className="text-gray-500 text-sm">C</span>
        </div>
      </div>

      {/* 💧 Humidity */}
      <div className="rounded-xl border border-gray-800 bg-gradient-to-br from-gray-800/40 to-gray-900/40 p-6 flex flex-col justify-between">
        <p className="text-sm text-gray-400 font-medium uppercase tracking-wider mb-2">Humidity</p>
        <div className="mt-2 flex items-baseline gap-2">
          <p className="text-3xl font-bold text-blue-400">
            {hum !== null ? `${hum.toFixed(1)}%` : '--'}
          </p>
        </div>
        {/* Progress bar */}
        <div className="mt-4 w-full bg-gray-800 rounded-full h-1.5">
          <div 
            className="bg-blue-500 h-1.5 rounded-full transition-all duration-500" 
            style={{ width: `${hum !== null ? Math.min(hum, 100) : 0}%` }}
          />
        </div>
      </div>

      {/* 🔥 Gas Level */}
      <div className="rounded-xl border border-gray-800 bg-gradient-to-br from-gray-800/40 to-gray-900/40 p-6 flex flex-col justify-between">
        <div className="flex justify-between items-start">
          <p className="text-sm text-gray-400 font-medium uppercase tracking-wider mb-2">Gas Level</p>
          <span className="text-xs text-gray-600 bg-gray-800 px-2 py-0.5 rounded-full">
            Raw: {gasRaw ?? '--'}
          </span>
        </div>
        <div className="mt-2 flex items-baseline gap-2">
          <p className="text-3xl font-bold text-white">
            {gasPct !== null ? `${gasPct}%` : '--'}
          </p>
        </div>
        {/* Progress bar */}
        <div className="mt-4 w-full bg-gray-800 rounded-full h-1.5">
          <div 
            className={`${gasColor} h-1.5 rounded-full transition-all duration-500`} 
            style={{ width: `${gasPct !== null ? Math.min(gasPct, 100) : 0}%` }}
          />
        </div>
      </div>

      {/* 📍 Location */}
      <div className="rounded-xl border border-gray-800 bg-gradient-to-br from-gray-800/40 to-gray-900/40 p-6 flex flex-col justify-between relative">
        <div className="flex justify-between items-start">
          <p className="text-sm text-gray-400 font-medium uppercase tracking-wider mb-2">Location</p>
          {reading.gps_simulated && (
            <span className="text-xs text-blue-400 bg-blue-900/30 px-2 py-0.5 rounded-full border border-blue-800/50">
              SIMULATED
            </span>
          )}
        </div>
        
        <div className="mt-2 text-sm text-gray-300">
          <div className="flex justify-between py-1">
            <span className="text-gray-500">Lat:</span>
            <span className="font-mono">{reading.gps_lat?.toFixed(5) ?? '--'}</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-gray-500">Lng:</span>
            <span className="font-mono">{reading.gps_lng?.toFixed(5) ?? '--'}</span>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${reading.gps_fix ? 'bg-green-500' : 'bg-gray-600'}`} />
          <span className="text-xs text-gray-500">
            {reading.gps_satellites !== null ? `${reading.gps_satellites} satellites` : 'No fix'}
          </span>
        </div>
      </div>

    </div>
  )
}
