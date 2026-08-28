"""
Financial Analyst node: 5-step multi-hop reasoning (async parallel)

Steps 1/2/3 run in parallel; Step 4 aggregates 1+2+3; Step 5 synthesizes cross-source risks.
Integrates technical analysis signals and peer industry comparison data.
"""

import asyncio
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.state import StockAnalysisState
from config import get_settings

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        s = get_settings()
        _llm = ChatOpenAI(model=s.openai_model, api_key=s.openai_api_key, temperature=0.1)
    return _llm

ANALYST_SYS_BASE = (
    "You are a top-tier sell-side research analyst specializing in fundamental analysis. "
    "Answer in English. Keep each analysis module to 3–5 concise, data-backed sentences."
)


async def _llm_async(system: str, user: str) -> tuple[str, dict]:
    """Returns (content, usage_metadata)."""
    try:
        resp = await _get_llm().ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        usage = resp.usage_metadata or {}
        return resp.content or "", usage
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Error during analysis: {str(e)}", {}


async def analyst_node(state: StockAnalysisState) -> dict:
    ticker             = state.get("ticker", "UNKNOWN")
    company            = state.get("company_name", ticker)
    metrics            = state.get("key_metrics") or {}
    price_data         = state.get("price_data") or {}
    financials         = state.get("financial_statements") or {}
    news               = state.get("recent_news") or []
    tech               = state.get("technical_analysis") or {}
    peers              = state.get("peer_comparison") or {}
    horizon            = state.get("horizon", "medium")
    events             = list(state.get("events") or [])
    reflection_critique = state.get("reflection_critique") or ""
    reflection_count   = state.get("reflection_count") or 0

    # Inject reflection critique into system prompt to guide deeper re-analysis
    if reflection_critique:
        ANALYST_SYS = (
            ANALYST_SYS_BASE
            + f"\n\n[Quality review from previous round found weaknesses — please strengthen these areas this round]\n{reflection_critique}"
        )
        events.append({
            "event_type": "agent_start",
            "node":       "analyst",
            "message":    f"Deepening analysis based on reflection (iteration {reflection_count}): {company} ({ticker})",
            "data":       {"steps": 5, "parallel": "1+2+3", "reflection_count": reflection_count},
        })
    else:
        ANALYST_SYS = ANALYST_SYS_BASE
        events.append({
            "event_type": "agent_start",
            "node":       "analyst",
            "message":    f"Starting multi-hop reasoning analysis: {company} ({ticker})",
            "data":       {"steps": 5, "parallel": "1+2+3"},
        })

    quarterly_rev = {}
    try:
        quarterly_rev = financials.get("quarterly_income", {}).get("Total Revenue", {})
    except Exception:
        pass

    tech_summary = tech.get("signal_summary", "Technical data unavailable")
    peer_summary = peers.get("summary", "Peer data unavailable")

    # ── Steps 1/2/3 in parallel ───────────────────────────────────
    s1_prompt = f"""Analyze {ticker}'s revenue profile:
Revenue Growth (YoY): {metrics.get('revenue_growth', 'N/A')}
Revenue TTM: ${metrics.get('revenue_ttm', 'N/A')}
Quarterly Revenue Data: {json.dumps(quarterly_rev, default=str)[:300]}
Technical Signal Reference: {tech_summary[:100]}

Assess: whether growth is accelerating/decelerating, growth quality, and competitive position in the industry."""

    s2_prompt = f"""Analyze {ticker}'s profitability quality:
Gross Margin: {metrics.get('gross_margin', 'N/A')}
Operating Margin: {metrics.get('operating_margin', 'N/A')}
Net Margin: {metrics.get('net_margin', 'N/A')}
ROE: {metrics.get('roe', 'N/A')}
Peer Comparison: {peer_summary[:100]}

Emphasize alignment between margin trends and revenue growth, and operating leverage effects."""

    s3_prompt = f"""Analyze {ticker}'s financial health:
Debt-to-Equity (D/E): {metrics.get('debt_to_equity', 'N/A')}
Current Ratio: {metrics.get('current_ratio', 'N/A')}
Free Cash Flow: ${metrics.get('free_cash_flow', 'N/A')}

Assess whether the balance sheet can support the growth story and its resilience in a downturn."""

    (s1, u1), (s2, u2), (s3, u3) = await asyncio.gather(
        _llm_async(ANALYST_SYS, s1_prompt),
        _llm_async(ANALYST_SYS, s2_prompt),
        _llm_async(ANALYST_SYS, s3_prompt),
    )

    for step_num, title, result in [
        (1, "Revenue Analysis", s1), (2, "Margin Analysis", s2), (3, "Financial Health", s3)
    ]:
        events.append({
            "event_type": "reasoning_step",
            "node": "analyst",
            "message": f"Step {step_num} complete: {title}",
            "data": {"step": step_num, "title": title, "finding": result[:150]},
        })

    # ── Step 4: Valuation (aggregates steps 1+2+3 + peer comparison) ──
    horizon_note = {
        "short": "(short-term: focus on technicals and near-term catalysts)",
        "long":  "(long-term: focus on DCF intrinsic value and competitive moat)",
    }.get(horizon, "(medium-term: balance technicals and fundamentals)")

    s4, u4 = await _llm_async(ANALYST_SYS, f"""Synthesize the first three steps and make a valuation judgment for {ticker} {horizon_note}:
Revenue: {s1[:100]}
Profitability: {s2[:100]}
Financial Health: {s3[:100]}

Valuation Metrics:
P/E (TTM): {metrics.get('pe_ratio','N/A')} | Forward P/E: {metrics.get('forward_pe','N/A')}
P/B: {metrics.get('pb_ratio','N/A')} | EV/EBITDA: {metrics.get('ev_ebitda','N/A')}
Current Price: ${price_data.get('current_price','N/A')} | Analyst Target: ${metrics.get('analyst_target_price','N/A')}
Peer Comparison: {peer_summary}
Technical Signal: {tech_summary[:120]}

Assess whether current valuation is reasonable and provide a target price range.""")

    events.append({
        "event_type": "reasoning_step",
        "node": "analyst",
        "message": "Step 4 complete: Valuation analysis",
        "data": {"step": 4, "title": "Valuation Analysis", "finding": s4[:150]},
    })

    # ── Step 5: Cross-source risk synthesis ───────────────────
    news_text = "\n".join(
        f"- [{n.get('published','')}] {n.get('title','')}" for n in news[:6]
    )

    s5, u5 = await _llm_async(ANALYST_SYS, f"""Synthesize cross-source risks for {ticker}:
Summary of first four steps:
Revenue={s1[:80]}...
Profitability={s2[:60]}...
Valuation={s4[:80]}...
Technical Signal: {tech.get('overall_signal','N/A')} — {tech.get('rsi_signal','N/A')}, {tech.get('macd_trend','N/A')}

Latest News:
{news_text[:600]}

Identify 3–5 key risks, each with data evidence. Focus on cross-source signal convergence.""")

    events.append({
        "event_type": "reasoning_step",
        "node": "analyst",
        "message": "Step 5 complete: Cross-source risk synthesis",
        "data": {"step": 5, "title": "Risk Synthesis", "finding": s5[:150]},
    })

    # Token summary
    _in  = sum(u.get("input_tokens",  0) for u in [u1, u2, u3, u4, u5])
    _out = sum(u.get("output_tokens", 0) for u in [u1, u2, u3, u4, u5])
    events.append({
        "event_type": "token_usage",
        "node": "analyst",
        "message": f"Tokens — input {_in:,} / output {_out:,} / total {_in + _out:,}",
        "data": {"input_tokens": _in, "output_tokens": _out, "total_tokens": _in + _out},
    })

    # Confidence score (total weight 4.0)
    data_quality = sum([
        1.0 if metrics    else 0.0,
        1.0 if price_data else 0.0,
        0.8 if financials else 0.0,
        0.5 if news       else 0.0,
        0.4 if tech.get("rsi_14") is not None else 0.0,
        0.3 if peers.get("peers") else 0.0,
    ])
    confidence = min(0.95, max(0.45, data_quality / 4.0))

    return {
        "revenue_analysis":       s1,
        "margin_analysis":        s2,
        "balance_sheet_analysis": s3,
        "valuation_analysis":     s4,
        "risk_synthesis":         s5,
        "confidence_score":       confidence,
        "analysis_complete":      True,
        "events":                 events,
    }
