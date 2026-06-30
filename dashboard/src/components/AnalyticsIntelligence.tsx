'use client'

import { useState } from 'react'

interface AnalyticsIntelligenceProps {
  deviceId: string
  summary: string
  onRefresh: () => void
}

export default function AnalyticsIntelligence({ deviceId, summary, onRefresh }: AnalyticsIntelligenceProps) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<string | null>(null)
  const [querying, setQuerying] = useState(false)

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim()) return

    setQuerying(true)
    setAnswer(null)

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const res = await fetch(`${apiBase}/api/analytics/${deviceId}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      if (!res.ok) throw new Error('Query request failed')
      const data = await res.json()
      setAnswer(data.answer)
    } catch (err) {
      console.error(err)
      setAnswer('Error: Failed to fetch query response from backend AI service.')
    } finally {
      setQuerying(false)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
      {/* 🤖 AI Operational Summary */}
      <div className="rounded-xl border border-gray-800 bg-gradient-to-br from-gray-900 to-black/60 p-6 flex flex-col justify-between">
        <div>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2 uppercase tracking-wider">
              <span className="text-violet-400">⚡</span> AI Operational Summary
            </h3>
            <button 
              onClick={onRefresh}
              className="text-xs text-gray-500 hover:text-violet-400 border border-gray-800 hover:border-violet-800 bg-gray-950 px-2 py-1 rounded transition-colors"
            >
              Refresh Summary
            </button>
          </div>
          <div className="prose prose-invert prose-sm text-gray-300 max-w-none leading-relaxed space-y-3 whitespace-pre-wrap">
            {summary || 'Loading AI summary...'}
          </div>
        </div>
        <div className="mt-4 text-[10px] text-gray-600">
          Summary generated asynchronously using Groq (llama-3.3-70b-versatile) on structured backend metrics.
        </div>
      </div>

      {/* 💬 Natural Language Operator Queries */}
      <div className="rounded-xl border border-gray-800 bg-gradient-to-br from-gray-900 to-black/60 p-6 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2 uppercase tracking-wider mb-4">
            <span className="text-cyan-400">💬</span> Operational Q&A Chat
          </h3>
          <form onSubmit={handleAsk} className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What was the average temperature yesterday?"
              className="flex-1 rounded-lg border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-white placeholder-gray-600 focus:border-violet-600 focus:outline-none focus:ring-1 focus:ring-violet-600"
              disabled={querying}
            />
            <button
              type="submit"
              className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-700 transition-colors disabled:bg-gray-800 disabled:text-gray-500"
              disabled={querying || !question.trim()}
            >
              {querying ? 'Asking...' : 'Ask AI'}
            </button>
          </form>

          {/* AI Response Container */}
          <div className="mt-6 min-h-[120px] rounded-lg border border-gray-800 bg-black/40 p-4">
            {querying && (
              <div className="flex flex-col gap-2 animate-pulse">
                <div className="h-4 bg-gray-800 rounded w-3/4" />
                <div className="h-4 bg-gray-800 rounded w-5/6" />
                <div className="h-4 bg-gray-800 rounded w-1/2" />
              </div>
            )}
            {!querying && !answer && (
              <p className="text-xs text-gray-500 italic text-center mt-8">
                Ask a natural language query about uptime, anomalies, environment stats, or confidence margins to receive answers from Groq.
              </p>
            )}
            {!querying && answer && (
              <div className="text-sm text-gray-300 leading-relaxed">
                <p className="font-semibold text-[10px] uppercase text-cyan-400 mb-1 tracking-wider">AI Response:</p>
                {answer}
              </div>
            )}
          </div>
        </div>
        <div className="mt-4 text-[10px] text-gray-600">
          Answers generated on-demand based on aggregated metrics context.
        </div>
      </div>
    </div>
  )
}
