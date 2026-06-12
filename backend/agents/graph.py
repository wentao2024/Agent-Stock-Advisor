"""
LangGraph 图

正常路径：
  START → planner → data_fetcher → technical_analyst → peer_benchmarker → analyst → synthesizer

置信度 >= 0.65 或已反思 2 次：
  synthesizer → END

置信度不足（< 0.65）且反思次数 < 2：
  synthesizer → reflector → analyst → synthesizer（循环）

节点职责：
- planner：解析 ticker，获取公司名称
- data_fetcher：并行拉取市场数据、财务三表、新闻、RAG（asyncio.gather）
- technical_analyst：RSI / MACD / 布林带 / 均线（纯量化，无 LLM）
- peer_benchmarker：同行业估值对比
- analyst：5步多跳推理（Step1/2/3 并行），支持反思建议注入
- synthesizer：结构化输出 StockRecommendation
- reflector：自我反思，生成针对性改进建议
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
from agents.nodes.reflector import reflector_node

logger = logging.getLogger(__name__)


def planner_node(state: StockAnalysisState) -> dict:
    """解析公司名称/ticker，获取标准化公司名，添加开始事件。"""
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


def route_after_synthesizer(state: StockAnalysisState) -> str:
    """
    置信度 >= 0.65 → 直接结束
    已反思 >= 2 次 → 强制结束（安全阀）
    否则 → 进入 reflector 触发深度重分析
    """
    rec        = state.get("recommendation") or {}
    confidence = rec.get("confidence", 1.0)
    ref_count  = state.get("reflection_count") or 0

    if confidence >= 0.65 or ref_count >= 2:
        logger.info(f"Route → END (confidence={confidence:.0%}, reflection_count={ref_count})")
        return "end"
    logger.info(f"Route → reflector (confidence={confidence:.0%}, reflection_count={ref_count})")
    return "reflect"


def build_graph(checkpointer=None):
    graph = StateGraph(StockAnalysisState)

    graph.add_node("planner",           planner_node)
    graph.add_node("data_fetcher",      data_fetcher_node)
    graph.add_node("technical_analyst", technical_analyst_node)
    graph.add_node("peer_benchmarker",  peer_benchmarker_node)
    graph.add_node("analyst",           analyst_node)
    graph.add_node("synthesizer",       synthesizer_node)
    graph.add_node("reflector",         reflector_node)

    graph.add_edge(START,               "planner")
    graph.add_edge("planner",           "data_fetcher")
    graph.add_edge("data_fetcher",      "technical_analyst")
    graph.add_edge("technical_analyst", "peer_benchmarker")
    graph.add_edge("peer_benchmarker",  "analyst")
    graph.add_edge("analyst",           "synthesizer")

    graph.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {"reflect": "reflector", "end": END},
    )
    graph.add_edge("reflector", "analyst")

    return graph.compile(checkpointer=checkpointer)
