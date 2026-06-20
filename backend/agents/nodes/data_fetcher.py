"""
Data Fetcher node — async parallel version

Uses asyncio.gather + asyncio.to_thread to call 4 yfinance tools in parallel,
compressing sequential ~8s latency down to ~2s (limited by the slowest single call).
"""
import asyncio
import logging

from agents.state import StockAnalysisState
from agents.tools.market_data import (
    get_key_metrics,
    get_stock_price_history,
    get_financial_statements,
    get_recent_news,
)

logger = logging.getLogger(__name__)


async def data_fetcher_node(state: StockAnalysisState) -> dict:
    ticker  = state.get("ticker", "AAPL")
    company = state.get("company_name", ticker)
    events  = list(state.get("events") or [])

    events.append({
        "event_type": "agent_start",
        "node":       "data_fetcher",
        "message":    f"Fetching {company} ({ticker}) market data in parallel (4 concurrent streams)...",
        "data":       {"ticker": ticker, "parallel": True},
    })

    # ── 4-way parallel: key metrics / price history / financials / news ──
    raw_metrics, raw_price, raw_financials, raw_news = await asyncio.gather(
        asyncio.to_thread(get_key_metrics.invoke,          {"ticker": ticker}),
        asyncio.to_thread(get_stock_price_history.invoke,  {"ticker": ticker, "period": "1y"}),
        asyncio.to_thread(get_financial_statements.invoke, {"ticker": ticker}),
        asyncio.to_thread(get_recent_news.invoke,          {"ticker": ticker, "max_items": 10}),
        return_exceptions=True,
    )

    # ── Process results (unified error fallback logic) ────────────
    def _safe_dict(r, label: str) -> dict:
        if isinstance(r, Exception):
            logger.error(f"{label} concurrent call error: {r}")
            return {}
        if isinstance(r, dict) and "error" in r:
            logger.warning(f"{label} returned error: {r['error']}")
            return {}
        return r or {}

    def _safe_list(r, label: str) -> list:
        if isinstance(r, Exception):
            logger.error(f"{label} concurrent call error: {r}")
            return []
        if isinstance(r, list) and r and "error" in r[0]:
            logger.warning(f"{label} returned error: {r[0]['error']}")
            return []
        return r or []

    key_metrics          = _safe_dict(raw_metrics,    "get_key_metrics")
    price_data           = _safe_dict(raw_price,      "get_stock_price_history")
    financial_statements = _safe_dict(raw_financials, "get_financial_statements")
    recent_news          = [
        n for n in _safe_list(raw_news, "get_recent_news")
        if n.get("title") and n.get("published", "").startswith("197") is False
    ]

    # ── Report after parallel fetch completes ─────────────────────
    if key_metrics:
        events.append({
            "event_type": "tool_result", "node": "data_fetcher",
            "message": (
                f"4-stream fetch complete — "
                f"PE={key_metrics.get('pe_ratio','N/A')} "
                f"Price=${price_data.get('current_price','N/A')} "
                f"News={len(recent_news)} items"
            ),
            "data": {},
        })
    else:
        events.append({
            "event_type": "tool_result", "node": "data_fetcher",
            "message": "Some data fetch failed, continuing with degraded data", "data": {},
        })

    return {
        "key_metrics":          key_metrics,
        "price_data":           price_data,
        "financial_statements": financial_statements,
        "recent_news":          recent_news,
        "data_fetch_complete":  True,
        "events":               events,
        "messages":             [],
    }
