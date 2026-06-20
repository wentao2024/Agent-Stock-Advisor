"""
Synthesizer node: with_structured_output → StockRecommendation
Integrates fundamental analysis + technical indicators + peer comparison into a complete structured report.
"""
import logging
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.state import StockAnalysisState
from models.schemas import DataSource, FinancialMetrics, StockRecommendation
from config import get_settings

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        s = get_settings()
        _llm = ChatOpenAI(model=s.openai_model, api_key=s.openai_api_key, temperature=0)
    return _llm


SYS = """You are a chief equity analyst at an investment bank. Generate a complete structured investment report based on the analysis below.

Requirements:
- recommendation: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
- confidence: float 0–1, reflecting analytical certainty
- one_line_thesis: core investment thesis (English, under 30 words)
- bull_case / bear_case: 3–5 items each, each backed by data
- key_risks: 3–5 major risks
- catalysts: near-term catalysts
- reasoning_chain: 5-step reasoning chain corresponding to the analysis process
- Target price derived from P/E/DCF reasoning and analyst consensus
- All text content must be in English
"""


async def synthesizer_node(state: StockAnalysisState) -> dict:
    ticker     = state.get("ticker", "UNKNOWN")
    company    = state.get("company_name", ticker)
    metrics    = state.get("key_metrics") or {}
    price_data = state.get("price_data") or {}
    news       = state.get("recent_news") or []
    tech       = state.get("technical_analysis") or {}
    peers      = state.get("peer_comparison") or {}
    horizon    = state.get("horizon", "medium")
    events     = list(state.get("events") or [])

    events.append({
        "event_type": "agent_start",
        "node": "synthesizer",
        "message": "Generating structured investment recommendation report...",
        "data": {},
    })

    # include_raw=True allows reading usage_metadata from the raw AIMessage
    structured_llm = _get_llm().with_structured_output(StockRecommendation, include_raw=True)

    current_price = price_data.get("current_price", 0.0) or 0.0

    # Technical analysis summary
    tech_section = ""
    if tech.get("rsi_14") is not None:
        tech_section = f"""
[Technical Analysis Signals]
Overall Signal: {tech.get('overall_signal','N/A')}
{tech.get('signal_summary', '')}
Bollinger Band Squeeze: {'Yes (volatility contracting, watch breakout direction)' if tech.get('bb_squeeze') else 'No'}
Price vs MA20: {tech.get('price_vs_ma20_pct', 'N/A')}%
"""

    # Peer comparison summary
    peer_section = ""
    if peers.get("peers"):
        peer_section = f"""
[Peer Comparison ({peers.get('sector','')} / {peers.get('industry','')})]
{peers.get('summary', '')}
Peer Avg P/E: {peers.get('peer_avg_pe','N/A')} | Avg Revenue Growth: {peers.get('peer_avg_revenue_growth','N/A')} | Avg Net Margin: {peers.get('peer_avg_net_margin','N/A')}
"""

    horizon_label = {"short": "Short-term (<3 months)", "long": "Long-term (>1 year)"}.get(horizon, "Medium-term (3–12 months)")

    prompt = f"""
Company: {company} ({ticker}) | Analysis Date: {date.today().isoformat()} | Current Price: ${current_price} | Investment Horizon: {horizon_label}

[Revenue Analysis] {state.get('revenue_analysis') or 'Insufficient data'}

[Margin Analysis] {state.get('margin_analysis') or 'Insufficient data'}

[Balance Sheet Analysis] {state.get('balance_sheet_analysis') or 'Insufficient data'}

[Valuation Analysis] {state.get('valuation_analysis') or 'Insufficient data'}

[Risk Synthesis] {state.get('risk_synthesis') or 'Insufficient data'}

[Key Financial Metrics]
P/E (TTM): {metrics.get('pe_ratio','N/A')} | Forward P/E: {metrics.get('forward_pe','N/A')} | P/B: {metrics.get('pb_ratio','N/A')}
Revenue Growth: {metrics.get('revenue_growth','N/A')} | Net Margin: {metrics.get('net_margin','N/A')}
Debt/Equity: {metrics.get('debt_to_equity','N/A')} | Free Cash Flow: ${metrics.get('free_cash_flow','N/A')}
Analyst Target Price: ${metrics.get('analyst_target_price','N/A')}
{tech_section}
{peer_section}
[Confidence Score] {state.get('confidence_score', 0.6):.0%}

[Latest News]
{chr(10).join(f'- {n.get("title","")}' for n in news[:4])}

Generate a complete structured investment recommendation report. All text content must be in English.
Set investment_horizon to: {horizon}
"""

    try:
        raw_result = await structured_llm.ainvoke([
            SystemMessage(content=SYS),
            HumanMessage(content=prompt),
        ])
        rec: StockRecommendation = raw_result["parsed"]
        if rec is None:
            raise ValueError(f"Structured output parsing failed: {raw_result.get('parsing_error')}")

        # Extract token usage
        raw_msg = raw_result.get("raw")
        usage   = (raw_msg.usage_metadata or {}) if raw_msg else {}
        _in  = usage.get("input_tokens",  0)
        _out = usage.get("output_tokens", 0)
        events.append({
            "event_type": "token_usage",
            "node": "synthesizer",
            "message": f"Tokens — input {_in:,} / output {_out:,} / total {_in + _out:,}",
            "data": {"input_tokens": _in, "output_tokens": _out, "total_tokens": _in + _out},
        })

        # Override LLM-generated numbers with actual data for accuracy
        rec.ticker            = ticker
        rec.company_name      = company
        rec.analysis_date     = date.today().isoformat()
        rec.current_price     = current_price
        rec.investment_horizon = horizon

        def _b(v, scale=1):
            try:
                return round(float(v) / scale, 2) if v is not None else None
            except Exception:
                return None

        rec.metrics = FinancialMetrics(
            current_price       = current_price,
            price_52w_high      = price_data.get("high_52w"),
            price_52w_low       = price_data.get("low_52w"),
            price_change_1y_pct = price_data.get("performance", {}).get("pct_1y"),
            pe_ratio            = metrics.get("pe_ratio"),
            forward_pe          = metrics.get("forward_pe"),
            pb_ratio            = metrics.get("pb_ratio"),
            ev_ebitda           = metrics.get("ev_ebitda"),
            market_cap_b        = _b(metrics.get("market_cap"), 1e9),
            revenue_growth_yoy  = metrics.get("revenue_growth"),
            earnings_growth_yoy = metrics.get("earnings_growth"),
            revenue_ttm_b       = _b(metrics.get("revenue_ttm"), 1e9),
            gross_margin        = metrics.get("gross_margin"),
            operating_margin    = metrics.get("operating_margin"),
            net_margin          = metrics.get("net_margin"),
            roe                 = metrics.get("roe"),
            debt_to_equity      = metrics.get("debt_to_equity"),
            current_ratio       = metrics.get("current_ratio"),
            free_cash_flow_b    = _b(metrics.get("free_cash_flow"), 1e9),
        )

        sources = [
            DataSource(source_type="yfinance", name=f"{ticker} Market Data", relevance="Price, financial metrics, news")
        ]
        if tech.get("rsi_14") is not None:
            sources.append(DataSource(
                source_type="yfinance",
                name=f"{ticker} Technical Indicators",
                relevance="RSI, MACD, Bollinger Bands, moving averages",
            ))
        if peers.get("peers"):
            sources.append(DataSource(
                source_type="yfinance",
                name=f"Peer Comparison ({peers.get('sector','')})",
                relevance="P/E, revenue growth, net margin benchmarking",
            ))
        if news:
            sources.append(DataSource(source_type="news", name="Latest News", relevance="Recent developments, sentiment signals"))
        rec.sources = sources

        rec_dict = rec.model_dump()

        events.append({
            "event_type": "recommendation",
            "node": "synthesizer",
            "message": f"Analysis complete: {rec.recommendation} (confidence {rec.confidence:.0%})",
            "data": {
                "recommendation": rec.recommendation,
                "confidence":     rec.confidence,
                "thesis":         rec.one_line_thesis,
            },
        })
        events.append({
            "event_type": "done",
            "node": "synthesizer",
            "message": "Report generated",
            "data": {},
        })

        return {"recommendation": rec_dict, "events": events}

    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        events.append({
            "event_type": "error", "node": "synthesizer",
            "message": str(e), "data": {},
        })
        return {"events": events, "error": str(e)}
