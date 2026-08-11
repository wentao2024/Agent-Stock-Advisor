"""
Stock Advisor API v4
- Input: company name (supports English names and tickers)
- LLM: OpenAI gpt-4o (sole provider)
- Short-term memory: Redis (LangGraph Checkpointer, supports resumable sessions)
- Long-term memory: PostgreSQL (historical analysis report persistence)
- Output: SSE streaming + downloadable txt report
"""

import contextlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, field_validator

import db
from agents.graph import build_graph
from config import get_settings
from ticker_resolver import company_name_to_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# Compiled LangGraph graph (initialized in lifespan)
_graph = None

# In-memory fallback cache (when PostgreSQL is unavailable)
_fallback_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 600


def _make_initial_state(ticker: str, horizon: str = "medium") -> dict:
    """Explicitly initialize all state fields to prevent KeyError."""
    return {
        "ticker":                 ticker,
        "company_name":           ticker,
        "horizon":                horizon,
        "messages":               [],
        "price_data":             {},
        "financial_statements":   {},
        "key_metrics":            {},
        "recent_news":            [],
        "technical_analysis":     {},
        "peer_comparison":        {},
        "revenue_analysis":       "",
        "margin_analysis":        "",
        "balance_sheet_analysis": "",
        "valuation_analysis":     "",
        "risk_synthesis":         "",
        "data_fetch_complete":    False,
        "analysis_complete":      False,
        "confidence_score":       0.0,
        "reflection_count":       0,
        "reflection_critique":    "",
        "recommendation":         None,
        "error":                  None,
        "events":                 [],
    }


def _make_config(thread_id: str) -> dict:
    return {
        "recursion_limit": 50,
        "configurable": {"thread_id": thread_id},
    }


def _get_graph():
    if _graph is None:
        raise HTTPException(503, "Service not ready, please try again later")
    return _graph


async def _save_report(ticker: str, company_name: str, horizon: str,
                       confidence: float, rec: dict) -> None:
    if db._pool:
        await db.save_report(ticker, company_name, horizon, confidence, rec)
    else:
        _fallback_cache[ticker] = (time.time(), rec)


async def _get_latest_report(ticker: str) -> Optional[dict]:
    if db._pool:
        return await db.get_latest_report(ticker, _CACHE_TTL)
    entry = _fallback_cache.get(ticker)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    _fallback_cache.pop(ticker, None)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("Stock Advisor (LangGraph + Redis Checkpointer + PostgreSQL) starting...")

    async with contextlib.AsyncExitStack() as stack:
        # ── Redis Checkpointer (optional) ────────────────────────
        checkpointer = None
        if settings.redis_url:
            try:
                from langgraph.checkpoint.redis.aio import AsyncRedisSaver
                checkpointer = await stack.enter_async_context(
                    AsyncRedisSaver.from_url(settings.redis_url)
                )
                await checkpointer.setup()
                logger.info("Redis Checkpointer enabled (resumable session support)")
            except Exception as e:
                logger.warning(f"Redis unavailable, running without persistence: {e}")
                checkpointer = None

        # Compile LangGraph graph
        _graph = build_graph(checkpointer)
        logger.info(f"LangGraph graph compiled (checkpointer={'Redis' if checkpointer else 'none'})")

        # ── PostgreSQL (optional) ────────────────────────────────
        if settings.database_url:
            try:
                await db.init_db(settings.database_url)
                logger.info("PostgreSQL connection pool ready (historical report persistence)")
            except Exception as e:
                logger.warning(f"PostgreSQL unavailable, using in-memory cache: {e}")

        # ── LangSmith tracing (optional) ─────────────────────────
        if settings.langchain_tracing_v2.lower() == "true" and settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"]    = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"]    = settings.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"]   = settings.langchain_endpoint
            logger.info(f"LangSmith tracing enabled: project={settings.langchain_project}")

        yield

    await db.close_db()
    logger.info("Service shut down")


app = FastAPI(
    title="Stock Advisor AI",
    description="LangGraph multi-agent stock analysis (Redis Checkpointer + PostgreSQL + LangSmith)",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


class AnalysisRequest(BaseModel):
    company_name: str
    horizon: str = "medium"

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: str) -> str:
        if v not in ("short", "medium", "long"):
            raise ValueError("horizon must be short, medium, or long")
        return v


# ── Health check ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model":        settings.openai_model,
        "checkpointer": "Redis" if settings.redis_url else "none",
        "persistence":  "PostgreSQL" if db._pool else "memory",
    }


@app.get("/api/resolve")
async def resolve(name: str = Query(...)):
    ticker = company_name_to_ticker(name)
    return {"input": name, "ticker": ticker}


# ── Streaming analysis SSE ────────────────────────────────────

@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalysisRequest):
    """SSE streaming analysis, pushes agent reasoning steps in real time."""
    graph   = _get_graph()
    ticker  = company_name_to_ticker(req.company_name)
    initial = _make_initial_state(ticker, req.horizon)

    async def gen():
        thread_id = str(uuid.uuid4())
        yield (
            f"data: {json.dumps({'event_type':'session_start','thread_id':thread_id,'ticker':ticker}, ensure_ascii=False)}\n\n"
        )

        emitted = 0
        last_rec = None
        last_rec_node = None
        try:
            async for chunk in graph.astream(initial, config=_make_config(thread_id)):
                for node_name, update in chunk.items():
                    all_events = update.get("events") or []
                    for ev in all_events[emitted:]:
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    emitted = len(all_events)

                    if update.get("recommendation"):
                        last_rec = update["recommendation"]
                        last_rec_node = node_name

            # Save final result only after graph completes, to avoid multiple writes during reflection loop
            if last_rec is not None:
                confidence = last_rec.get("confidence", 0.0)
                await _save_report(
                    ticker, req.company_name, req.horizon, confidence, last_rec
                )
                yield (
                    f"data: {json.dumps({'event_type':'final_recommendation','node':last_rec_node,'message':'Analysis complete','data':last_rec}, ensure_ascii=False)}\n\n"
                )
        except Exception as e:
            logger.error(f"Graph execution error: {e}", exc_info=True)
            yield f"data: {json.dumps({'event_type':'error','node':'graph','message':str(e),'data':{}}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Non-streaming analysis ────────────────────────────────────

@app.post("/api/analyze")
async def analyze(req: AnalysisRequest):
    """Non-streaming analysis (for testing/batch processing)."""
    graph     = _get_graph()
    ticker    = company_name_to_ticker(req.company_name)
    initial   = _make_initial_state(ticker, req.horizon)
    thread_id = str(uuid.uuid4())
    try:
        final = await graph.ainvoke(initial, config=_make_config(thread_id))
        rec   = final.get("recommendation")
        if rec:
            await _save_report(
                ticker, req.company_name, req.horizon,
                rec.get("confidence", 0.0), rec
            )
        return {
            "company_name":   req.company_name,
            "ticker":         ticker,
            "thread_id":      thread_id,
            "recommendation": rec,
            "error":          final.get("error"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── txt report download ───────────────────────────────────────

@app.post("/api/analyze/download")
async def analyze_download(req: AnalysisRequest):
    """Returns a downloadable txt report; reuses cached analysis within 10 minutes to avoid redundant calls."""
    graph  = _get_graph()
    ticker = company_name_to_ticker(req.company_name)

    rec = await _get_latest_report(ticker)

    if rec is None:
        initial   = _make_initial_state(ticker, req.horizon)
        thread_id = str(uuid.uuid4())
        try:
            final = await graph.ainvoke(initial, config=_make_config(thread_id))
            rec   = final.get("recommendation")
            if not rec:
                raise HTTPException(500, f"Analysis failed: {final.get('error','unknown error')}")
            await _save_report(
                ticker, req.company_name, req.horizon,
                rec.get("confidence", 0.0), rec
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))

    txt      = _format_txt(rec, req.company_name, ticker)
    filename = f"{ticker}_analysis_{date.today().isoformat()}.txt"
    return PlainTextResponse(
        content=txt,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/plain; charset=utf-8",
        },
    )


# ── Historical analysis records ───────────────────────────────

@app.get("/api/reports/{ticker}")
async def get_reports(ticker: str, limit: int = Query(10, ge=1, le=50)):
    """Retrieve historical analysis records for a ticker (only available in PostgreSQL mode)."""
    ticker = ticker.upper()
    history = await db.get_report_history(ticker, limit)
    return {"ticker": ticker, "count": len(history), "reports": history}


def _format_txt(rec: dict, company_name: str, ticker: str) -> str:
    SEP = "=" * 64

    def _pct(v):
        if v is None: return "N/A"
        try: return f"{float(v)*100:.1f}%"
        except: return "N/A"

    def _usd_b(v):
        if v is None: return "N/A"
        try: return f"${float(v):.1f}B"
        except: return "N/A"

    lines = [
        SEP,
        "  Stock Analysis Report — AI Generated",
        f"  {rec.get('company_name', company_name)} ({ticker})",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Model: {settings.openai_model}",
        SEP, "",
        f"Recommendation:    {rec.get('recommendation','N/A')}",
        f"Confidence:        {rec.get('confidence',0):.0%}",
        f"Investment Horizon:{rec.get('investment_horizon','medium')}",
        "",
        "Core Investment Thesis",
        f"  {rec.get('one_line_thesis','')}",
        "",
        "Price Information",
        f"  Current Price: ${rec.get('current_price',0):.2f}",
    ]

    if rec.get("target_price_low") and rec.get("target_price_high"):
        lines.append(f"  Target Price Range: ${rec['target_price_low']:.0f} – ${rec['target_price_high']:.0f}")
    if rec.get("upside_pct") is not None:
        lines.append(f"  Upside/Downside: {rec['upside_pct']:+.1f}%")

    lines += ["", "Bull Case (Reasons to Buy)"]
    for b in rec.get("bull_case", []):
        lines.append(f"  ✓ {b}")

    lines += ["", "Bear Case (Risk Factors)"]
    for b in rec.get("bear_case", []):
        lines.append(f"  ✗ {b}")

    lines += ["", "Key Risks"]
    for r in rec.get("key_risks", []):
        lines.append(f"  ⚠ {r}")

    if rec.get("catalysts"):
        lines += ["", "Near-term Catalysts"]
        for c in rec.get("catalysts", []):
            lines.append(f"  ★ {c}")

    m = rec.get("metrics", {})
    if m:
        lines += [
            "", "Key Financial Metrics",
            f"  Market Cap:       {_usd_b(m.get('market_cap_b'))}",
            f"  Revenue (TTM):    {_usd_b(m.get('revenue_ttm_b'))}",
            f"  Revenue Growth:   {_pct(m.get('revenue_growth_yoy'))}",
            f"  Gross Margin:     {_pct(m.get('gross_margin'))}",
            f"  Net Margin:       {_pct(m.get('net_margin'))}",
            f"  P/E (TTM):        {m.get('pe_ratio','N/A')}",
            f"  Forward P/E:      {m.get('forward_pe','N/A')}",
            f"  Debt/Equity:      {m.get('debt_to_equity','N/A')}",
            f"  Free Cash Flow:   {_usd_b(m.get('free_cash_flow_b'))}",
            f"  52-Week High:     ${m.get('price_52w_high') or 0:.2f}",
            f"  52-Week Low:      ${m.get('price_52w_low') or 0:.2f}",
        ]

    chain = rec.get("reasoning_chain", [])
    if chain:
        lines += ["", "Multi-hop Reasoning Chain"]
        for s in chain:
            lines += [
                f"\n  Step {s.get('step','?')}: {s.get('title','')}",
                f"    {s.get('finding','')}",
            ]
            if s.get("implication"):
                lines.append(f"    → {s.get('implication','')}")

    sources = rec.get("sources", [])
    if sources:
        lines += ["", "Data Sources"]
        for src in sources:
            lines.append(f"  [{src.get('source_type','').upper()}] {src.get('name','')} — {src.get('relevance','')}")

    lines += [
        "", SEP,
        "Disclaimer: This report is AI-generated for research purposes only and does not constitute investment advice.",
        "Please consult a qualified financial advisor before making investment decisions.",
        SEP,
    ]
    return "\n".join(lines)
