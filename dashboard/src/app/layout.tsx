import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'SentinelEdge — Edge AI Fault Detection',
  description:
    'Industrial edge intelligence platform: ESP32 TinyML vibration fault detection with real-time GPS-tagged anomaly dashboard.',
  keywords: ['edge AI', 'TinyML', 'IoT', 'fault detection', 'ESP32', 'industrial monitoring'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className={`${inter.className} bg-gray-950 text-gray-100 min-h-screen`}>
        {/* Global Navigation */}
        <nav className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-14">
              <div className="flex items-center gap-3">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center">
                  <span className="text-xs font-bold text-white">SE</span>
                </div>
                <span className="font-bold text-white tracking-tight">SentinelEdge</span>
                <span className="hidden sm:block text-xs text-gray-500 font-mono border border-gray-700 rounded px-1.5 py-0.5">
                  Edge AI v1.0
                </span>
              </div>
              <div className="flex items-center gap-1">
                {[
                  { href: '/dashboard',  label: 'Dashboard' },
                  { href: '/anomalies', label: 'Anomalies' },
                  { href: '/map',       label: 'Map' },
                  { href: '/analytics', label: 'Analytics' },
                ].map(({ href, label }) => (
                  <a
                    key={href}
                    href={href}
                    className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded-md transition-colors"
                  >
                    {label}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
      </body>
    </html>
  )
}
