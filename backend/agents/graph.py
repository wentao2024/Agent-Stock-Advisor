"""
LangGraph graph

Normal path:
  START → planner → data_fetcher → technical_analyst → peer_benchmarker → analyst → synthesizer

Confidence >= 0.65 or already reflected 2 times:
  synthesizer → END

Confidence < 0.65 and reflection count < 2:
  synthesizer → reflector → analyst → synthesizer (loop)

Node responsibilities:
- planner: resolve ticker, fetch company name
- data_fetcher: parallel fetch of market data, financial statements, news (asyncio.gather)
- technical_analyst: RSI / MACD / Bollinger Bands / moving averages (pure quant, no LLM)
- peer_benchmarker: peer industry valuation comparison
- analyst: 5-step multi-hop reasoning (Steps 1/2/3 in parallel), supports reflection injection
- synthesizer: structured output StockRecommendation
- reflector: self-reflection, generates targeted improvement suggestions
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
    """Resolve company name/ticker, fetch canonical company name, emit start event."""
    from ticker_resolver import company_name_to_ticker

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
        "message": f"Starting analysis: {company_name} ({ticker})",
        "data": {"ticker": ticker, "company": company_name},
    })

    return {
        "ticker":       ticker,
        "company_name": company_name,
        "events":       events,
    }


def route_after_synthesizer(state: StockAnalysisState) -> str:
    """
    Confidence >= 0.65 → end directly
    Reflection count >= 2 → force end (safety valve)
    Otherwise → enter reflector for deep re-analysis
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
