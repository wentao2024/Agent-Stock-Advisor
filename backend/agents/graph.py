"""
LangGraph 图

START → planner → data_fetcher → technical_analyst → peer_benchmarker → analyst → synthesizer → END

- planner：解析 ticker，获取公司名称，不重置其他 state
- data_fetcher：市场数据、财务三表、新闻、RAG
- technical_analyst：RSI / MACD / 布林带 / 均线（纯量化，无 LLM）
- peer_benchmarker：同行业估值对比
- analyst：5步多跳推理（Step1/2/3 并行）
- synthesizer：结构化输出 StockRecommendation
"""
import logging

import yfinance as yf
from langgraph.graph import END, START, StateGraph

from agents.state import StockAnalysisState
from agents.nodes.data_fetcher import data_fetcher_node
from agents.nodes.technical_analyst import technical_analyst_node
from agents.nodes.peer_benchmarker import peer_benchmarker_node
from agents.nodes.analyst import analyst_node
from agents.nodes.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


def planner_node(state: StockAnalysisState) -> dict:
    """解析公司名称/ticker，获取标准化公司名，添加开始事件。
    不重置其他 state 字段（已由 _make_initial_state 初始化）。
    """
    from rag.indexer import company_name_to_ticker

    raw    = state.get("ticker") or state.get("company_name") or "AAPL"
    ticker = company_name_to_ticker(raw)

    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName", ticker)
    except Exception:
        company_name = ticker

    events = list(state.get("events") or [])
    events.append({
        "event_type": "agent_start",
        "node": "planner",
        "message": f"开始分析：{company_name}（{ticker}）",
        "data": {"ticker": ticker, "company": company_name},
    })

    return {
        "ticker":       ticker,
        "company_name": company_name,
        "events":       events,
    }


def build_graph():
    graph = StateGraph(StockAnalysisState)

    graph.add_node("planner",           planner_node)
    graph.add_node("data_fetcher",      data_fetcher_node)
    graph.add_node("technical_analyst", technical_analyst_node)
    graph.add_node("peer_benchmarker",  peer_benchmarker_node)
    graph.add_node("analyst",           analyst_node)
    graph.add_node("synthesizer",       synthesizer_node)

    graph.add_edge(START,              "planner")
    graph.add_edge("planner",          "data_fetcher")
    graph.add_edge("data_fetcher",     "technical_analyst")
    graph.add_edge("technical_analyst","peer_benchmarker")
    graph.add_edge("peer_benchmarker", "analyst")
    graph.add_edge("analyst",          "synthesizer")
    graph.add_edge("synthesizer",      END)

    return graph.compile()


compiled_graph = build_graph()
