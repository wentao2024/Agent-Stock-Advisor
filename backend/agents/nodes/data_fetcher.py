"""
Data Fetcher 节点 — 直接函数调用方式

直接调用 Python 函数收集数据，无 LLM 工具调用，
彻底避免消息状态累积导致的 NameError/KeyError。
"""
import logging

from agents.state import StockAnalysisState
from agents.tools.market_data import (
    get_key_metrics,
    get_stock_price_history,
    get_financial_statements,
    get_recent_news,
)
from agents.tools.rag_tool import search_financial_knowledge_base

logger = logging.getLogger(__name__)


def data_fetcher_node(state: StockAnalysisState) -> dict:
    ticker     = state.get("ticker", "AAPL")
    company    = state.get("company_name", ticker)
    include_rag = state.get("include_rag", True)
    events     = list(state.get("events") or [])

    events.append({
        "event_type": "agent_start",
        "node": "data_fetcher",
        "message": f"正在收集 {company}（{ticker}）的市场数据和财务指标...",
        "data": {"ticker": ticker},
    })

    # 1. 核心财务指标
    events.append({
        "event_type": "tool_call", "node": "data_fetcher",
        "message": "调用工具：get_key_metrics", "data": {},
    })
    try:
        key_metrics = get_key_metrics.invoke({"ticker": ticker})
        if "error" in key_metrics:
            logger.warning(f"get_key_metrics 错误: {key_metrics.get('error')}")
            key_metrics = {}
        else:
            events.append({
                "event_type": "tool_result", "node": "data_fetcher",
                "message": f"财务指标获取成功：PE={key_metrics.get('pe_ratio','N/A')} 营收增长={key_metrics.get('revenue_growth','N/A')}",
                "data": {},
            })
    except Exception as e:
        logger.error(f"get_key_metrics 失败: {e}")
        key_metrics = {}

    # 2. 股价历史
    events.append({
        "event_type": "tool_call", "node": "data_fetcher",
        "message": "调用工具：get_stock_price_history", "data": {},
    })
    try:
        price_data = get_stock_price_history.invoke({"ticker": ticker, "period": "1y"})
        if "error" in price_data:
            logger.warning(f"get_stock_price_history 错误: {price_data.get('error')}")
            price_data = {}
        else:
            events.append({
                "event_type": "tool_result", "node": "data_fetcher",
                "message": f"股价数据获取成功：当前价 ${price_data.get('current_price','N/A')}",
                "data": {},
            })
    except Exception as e:
        logger.error(f"get_stock_price_history 失败: {e}")
        price_data = {}

    # 3. 财务三表
    events.append({
        "event_type": "tool_call", "node": "data_fetcher",
        "message": "调用工具：get_financial_statements", "data": {},
    })
    try:
        financial_statements = get_financial_statements.invoke({"ticker": ticker})
        if "error" in financial_statements:
            logger.warning(f"get_financial_statements 错误: {financial_statements.get('error')}")
            financial_statements = {}
        else:
            events.append({
                "event_type": "tool_result", "node": "data_fetcher",
                "message": "财务三表获取成功（损益表/资产负债表/现金流量表）",
                "data": {},
            })
    except Exception as e:
        logger.error(f"get_financial_statements 失败: {e}")
        financial_statements = {}

    # 4. 最新新闻
    events.append({
        "event_type": "tool_call", "node": "data_fetcher",
        "message": "调用工具：get_recent_news", "data": {},
    })
    try:
        recent_news = get_recent_news.invoke({"ticker": ticker, "max_items": 10})
        if isinstance(recent_news, list) and recent_news and "error" in recent_news[0]:
            recent_news = []
        else:
            events.append({
                "event_type": "tool_result", "node": "data_fetcher",
                "message": f"新闻获取成功：{len(recent_news)} 条最新报道",
                "data": {},
            })
    except Exception as e:
        logger.error(f"get_recent_news 失败: {e}")
        recent_news = []

    # 5. ChromaDB RAG 检索（可选）
    rag_contexts = list(state.get("rag_contexts") or [])
    if include_rag:
        events.append({
            "event_type": "tool_call", "node": "data_fetcher",
            "message": "调用工具：search_financial_knowledge_base（ChromaDB）",
            "data": {},
        })
        for query in ["risk factors supply chain competition", "revenue growth strategy"]:
            try:
                result = search_financial_knowledge_base.invoke({
                    "query": query, "ticker": ticker
                })
                if result and "not initialized" not in result and "No relevant" not in result:
                    rag_contexts.append(result)
                    events.append({
                        "event_type": "tool_result", "node": "data_fetcher",
                        "message": "ChromaDB RAG 检索完成",
                        "data": {},
                    })
                    break
            except Exception as e:
                logger.warning(f"RAG 检索失败: {e}")

    return {
        "key_metrics":          key_metrics,
        "price_data":           price_data,
        "financial_statements": financial_statements,
        "recent_news":          recent_news,
        "rag_contexts":         rag_contexts,
        "data_fetch_complete":  True,
        "events":               events,
        "messages":             [],
    }


