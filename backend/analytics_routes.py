"""
SentinelEdge — Analytics API Routes
==================================
Endpoints for fetching aggregated device analytics and querying them via Groq.
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import analytics_service
from groq_client import get_groq_client
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# In-memory cache for operational summaries
# Key: device_id, Value: (metrics_hash, summary_markdown)
_summary_cache: Dict[str, tuple] = {}

class QueryRequest(BaseModel):
    question: str

async def generate_groq_summary(device_id: str, metrics: Dict[str, Any]) -> str:
    """Generate operational summary using Groq based strictly on metrics JSON."""
    prompt = f"""You are SentinelEdge AI, an advanced industrial analytics assistant. 
Review the following operational telemetry metrics for device '{device_id}' over the last 24 hours:

{json.dumps(metrics, indent=2)}

Generate a concise, professional operational report in Markdown format.
Highlight any anomalies or low-confidence indicators in bold.
Include:
1. Equipment Uptime & Health Assessment.
2. Distribution of known states and transition observations.
3. Environmental summaries (temperature, humidity averages).
4. Shadow mode unknown behavior alerts (if any candidates were flagged).

Keep the summary under 180 words. Write directly and professionally. Do not include raw JSON.
"""
    try:
        client = get_groq_client()
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate summary from Groq: {e}")
        return "Error: Could not generate AI operational summary at this time."

async def generate_groq_query_answer(device_id: str, metrics: Dict[str, Any], question: str) -> str:
    """Answer natural language queries using Groq based strictly on metrics JSON."""
    prompt = f"""You are SentinelEdge AI, an industrial analytics advisor.
Here are the aggregated operational metrics for device '{device_id}' over the last 24 hours:

{json.dumps(metrics, indent=2)}

The user asks: "{question}"

Answer the question directly, concisely, and conversationally based strictly on the provided metrics.
If the metrics do not contain the answer, say "The current 24-hour metrics do not contain that information."
Do not speculate or hallucinate. Keep the response under 120 words.
"""
    try:
        client = get_groq_client()
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate query answer from Groq: {e}")
        return "Error: Could not process query at this time."


@router.get("/{device_id}/summary")
async def get_device_summary(device_id: str, range_hours: int = 24):
    """
    Get aggregated analytics metrics and an AI-generated operational summary.
    Employs SHA-256 caching of the metrics to prevent unnecessary LLM invocation.
    """
    # 1. Fetch metrics
    metrics = analytics_service.get_analytics_summary_json(device_id, range_hours)
    
    # 2. Compute stable hash of metrics JSON (excluding range_hours metadata to keep it pure)
    metrics_str = json.dumps(metrics, sort_keys=True)
    metrics_hash = hashlib.sha256(metrics_str.encode("utf-8")).hexdigest()
    
    # 3. Check cache
    cached_hash, cached_summary = _summary_cache.get(device_id, (None, None))
    if cached_hash == metrics_hash and cached_summary:
        logger.debug(f"Cache HIT for device summary: {device_id}")
        return {
            "metrics": metrics,
            "summary": cached_summary,
            "cache_hit": True
        }
    
    # 4. Cache miss: generate summary and populate cache
    logger.info(f"Cache MISS for device summary: {device_id}. Invoking Groq...")
    summary = await generate_groq_summary(device_id, metrics)
    _summary_cache[device_id] = (metrics_hash, summary)
    
    return {
        "metrics": metrics,
        "summary": summary,
        "cache_hit": False
    }


@router.post("/{device_id}/query")
async def query_device_metrics(device_id: str, request: QueryRequest, range_hours: int = 24):
    """
    Query the device's operational metrics using natural language.
    """
    metrics = analytics_service.get_analytics_summary_json(device_id, range_hours)
    answer = await generate_groq_query_answer(device_id, metrics, request.question)
    return {
        "device_id": device_id,
        "question": request.question,
        "answer": answer
    }
