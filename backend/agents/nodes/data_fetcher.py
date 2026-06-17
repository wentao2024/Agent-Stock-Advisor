"""
Data Fetcher 节点 — 异步并行版

使用 asyncio.gather + asyncio.to_thread 并行调用 4 个 yfinance 工具，
将顺序执行的 ~8s 延迟压缩到 ~2s（取最慢单次调用时间）。
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
        "message":    f"并行拉取 {company}（{ticker}）市场数据（4 路并发）...",
        "data":       {"ticker": ticker, "parallel": True},
    })

    # ── 4 路并行：核心指标 / 股价历史 / 财务三表 / 新闻 ──────────
    raw_metrics, raw_price, raw_financials, raw_news = await asyncio.gather(
        asyncio.to_thread(get_key_metrics.invoke,          {"ticker": ticker}),
        asyncio.to_thread(get_stock_price_history.invoke,  {"ticker": ticker, "period": "1y"}),
        asyncio.to_thread(get_financial_statements.invoke, {"ticker": ticker}),
        asyncio.to_thread(get_recent_news.invoke,          {"ticker": ticker, "max_items": 10}),
        return_exceptions=True,
    )

    # ── 处理结果（统一错误降级逻辑）────────────────────────────
    def _safe_dict(r, label: str) -> dict:
        if isinstance(r, Exception):
            logger.error(f"{label} 并发调用异常: {r}")
            return {}
        if isinstance(r, dict) and "error" in r:
            logger.warning(f"{label} 返回错误: {r['error']}")
            return {}
        return r or {}

    def _safe_list(r, label: str) -> list:
        if isinstance(r, Exception):
            logger.error(f"{label} 并发调用异常: {r}")
            return []
        if isinstance(r, list) and r and "error" in r[0]:
            logger.warning(f"{label} 返回错误: {r[0]['error']}")
            return []
        return r or []

    key_metrics          = _safe_dict(raw_metrics,    "get_key_metrics")
    price_data           = _safe_dict(raw_price,      "get_stock_price_history")
    financial_statements = _safe_dict(raw_financials, "get_financial_statements")
    recent_news          = [
        n for n in _safe_list(raw_news, "get_recent_news")
        if n.get("title") and n.get("published", "").startswith("197") is False
    ]

    # ── 并发完成后汇报 ──────────────────────────────────────────
    if key_metrics:
        events.append({
            "event_type": "tool_result", "node": "data_fetcher",
            "message": (
                f"4路数据拉取完成 — "
                f"PE={key_metrics.get('pe_ratio','N/A')} "
                f"股价=${price_data.get('current_price','N/A')} "
                f"新闻{len(recent_news)}条"
            ),
            "data": {},
        })
    else:
        events.append({
            "event_type": "tool_result", "node": "data_fetcher",
            "message": "部分数据拉取失败，降级继续分析", "data": {},
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
