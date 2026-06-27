'use client'

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { FAULT_COLORS } from '@/lib/supabase'

interface DataPoint {
  date:        string
  normal:      number
  imbalance:   number
  obstruction: number
  loose_mount: number
}

interface Props {
  data: DataPoint[]
}

export default function AnalyticsChart({ data }: Props) {
  if (!data.length) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-600 text-sm">
        No data in selected time range
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="date"
          tick={{ fill: '#6b7280', fontSize: 11 }}
          axisLine={{ stroke: '#374151' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#6b7280', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          allowDecimals={false}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#111827',
            border: '1px solid #374151',
            borderRadius: '8px',
            fontSize: '12px',
            color: '#f9fafb',
          }}
          cursor={{ fill: 'rgba(255,255,255,0.05)' }}
        />
        <Legend
          wrapperStyle={{ fontSize: '12px', color: '#9ca3af', paddingTop: '12px' }}
        />
        <Bar dataKey="imbalance"   name="Imbalance"   fill={FAULT_COLORS[1]} radius={[2,2,0,0]} />
        <Bar dataKey="obstruction" name="Obstruction" fill={FAULT_COLORS[2]} radius={[2,2,0,0]} />
        <Bar dataKey="loose_mount" name="Loose Mount" fill={FAULT_COLORS[3]} radius={[2,2,0,0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
