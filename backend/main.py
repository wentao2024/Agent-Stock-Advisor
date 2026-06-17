"""
Stock Advisor API v4
- 输入：公司名称（支持中英文、ticker 均可）
- LLM：OpenAI gpt-4o（唯一提供商）
- 短期记忆：Redis（LangGraph Checkpointer，支持断点续传）
- 长期记忆：PostgreSQL（历史分析报告持久化）
- 输出：SSE 流式 + 可下载 txt 报告（全中文）
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

# 编译好的 LangGraph 图（lifespan 中初始化）
_graph = None

# 内存缓存（PostgreSQL 不可用时的降级方案）
_fallback_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 600


def _make_initial_state(ticker: str, horizon: str = "medium") -> dict:
    """显式初始化所有 state 字段，彻底防止 KeyError"""
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
        raise HTTPException(503, "服务未就绪，请稍后重试")
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
    logger.info("Stock Advisor（LangGraph + Redis Checkpointer + PostgreSQL）启动中...")

    async with contextlib.AsyncExitStack() as stack:
        # ── Redis Checkpointer（可选）────────────────────────────
        checkpointer = None
        if settings.redis_url:
            try:
                from langgraph.checkpoint.redis.aio import AsyncRedisSaver
                checkpointer = await stack.enter_async_context(
                    AsyncRedisSaver.from_url(settings.redis_url)
                )
                await checkpointer.setup()
                logger.info("Redis Checkpointer 已启用（断点续传支持）")
            except Exception as e:
                logger.warning(f"Redis 不可用，以无持久化模式运行: {e}")
                checkpointer = None

        # 编译 LangGraph 图
        _graph = build_graph(checkpointer)
        logger.info(f"LangGraph 图已编译（checkpointer={'Redis' if checkpointer else '无'}）")

        # ── PostgreSQL（可选）────────────────────────────────────
        if settings.database_url:
            try:
                await db.init_db(settings.database_url)
                logger.info("PostgreSQL 连接池已就绪（历史报告持久化）")
            except Exception as e:
                logger.warning(f"PostgreSQL 不可用，使用内存缓存: {e}")

        # ── LangSmith 追踪（可选）───────────────────────────────
        if settings.langchain_tracing_v2.lower() == "true" and settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"]    = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"]    = settings.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"]   = settings.langchain_endpoint
            logger.info(f"LangSmith tracing 已启用：project={settings.langchain_project}")

        yield

    await db.close_db()
    logger.info("服务已关闭")


app = FastAPI(
    title="Stock Advisor AI",
    description="LangGraph 多 Agent 股票分析（Redis Checkpointer + PostgreSQL + LangSmith）",
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
            raise ValueError("horizon 必须是 short、medium 或 long")
        return v


# ── 健康检查 ──────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "正常",
        "model":        settings.openai_model,
        "checkpointer": "Redis" if settings.redis_url else "无",
        "persistence":  "PostgreSQL" if db._pool else "内存",
    }


@app.get("/api/resolve")
async def resolve(name: str = Query(...)):
    ticker = company_name_to_ticker(name)
    return {"input": name, "ticker": ticker}


# ── 流式分析 SSE ──────────────────────────────────────────────

@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalysisRequest):
    """SSE 流式分析，实时推送 Agent 推理过程"""
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

            # 只在图执行完毕后保存最终结果，避免反思循环中多次写库
            if last_rec is not None:
                confidence = last_rec.get("confidence", 0.0)
                await _save_report(
                    ticker, req.company_name, req.horizon, confidence, last_rec
                )
                yield (
                    f"data: {json.dumps({'event_type':'final_recommendation','node':last_rec_node,'message':'分析完成','data':last_rec}, ensure_ascii=False)}\n\n"
                )
        except Exception as e:
            logger.error(f"图执行错误：{e}", exc_info=True)
            yield f"data: {json.dumps({'event_type':'error','node':'graph','message':str(e),'data':{}}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 非流式分析 ────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(req: AnalysisRequest):
    """非流式分析（测试/批量处理用）"""
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


# ── txt 报告下载 ──────────────────────────────────────────────

@app.post("/api/analyze/download")
async def analyze_download(req: AnalysisRequest):
    """返回可下载的中文 txt 报告；优先复用 10 分钟内的分析缓存，避免重复分析"""
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
                raise HTTPException(500, f"分析失败：{final.get('error','未知错误')}")
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
    from urllib.parse import quote as _url_quote
    filename_cn = f"{ticker}_分析报告_{date.today().isoformat()}.txt"
    filename_encoded = _url_quote(filename_cn, safe="")
    return PlainTextResponse(
        content=txt,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"; '
                f"filename*=UTF-8''{filename_encoded}"
            ),
            "Content-Type": "text/plain; charset=utf-8",
        },
    )


# ── 历史分析记录 ──────────────────────────────────────────────

@app.get("/api/reports/{ticker}")
async def get_reports(ticker: str, limit: int = Query(10, ge=1, le=50)):
    """查询某 ticker 的历史分析记录（仅 PostgreSQL 模式下有数据）"""
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
        "  股票分析报告 — AI 生成",
        f"  {rec.get('company_name', company_name)}（{ticker}）",
        f"  生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 模型：{settings.openai_model}",
        SEP, "",
        f"投资建议：{rec.get('recommendation','N/A')}",
        f"置  信  度：{rec.get('confidence',0):.0%}",
        f"投资期限：{rec.get('investment_horizon','medium')}",
        "",
        "核心投资论点",
        f"  {rec.get('one_line_thesis','')}",
        "",
        "价格信息",
        f"  当前价格：${rec.get('current_price',0):.2f}",
    ]

    if rec.get("target_price_low") and rec.get("target_price_high"):
        lines.append(f"  目标价区间：${rec['target_price_low']:.0f} ~ ${rec['target_price_high']:.0f}")
    if rec.get("upside_pct") is not None:
        lines.append(f"  潜在涨跌幅：{rec['upside_pct']:+.1f}%")

    lines += ["", "看多理由（买入依据）"]
    for b in rec.get("bull_case", []):
        lines.append(f"  ✓ {b}")

    lines += ["", "看空理由（风险因素）"]
    for b in rec.get("bear_case", []):
        lines.append(f"  ✗ {b}")

    lines += ["", "关键风险"]
    for r in rec.get("key_risks", []):
        lines.append(f"  ⚠ {r}")

    if rec.get("catalysts"):
        lines += ["", "近期催化剂"]
        for c in rec.get("catalysts", []):
            lines.append(f"  ★ {c}")

    m = rec.get("metrics", {})
    if m:
        lines += [
            "", "财务核心指标",
            f"  市值：          {_usd_b(m.get('market_cap_b'))}",
            f"  TTM 营收：      {_usd_b(m.get('revenue_ttm_b'))}",
            f"  营收增长率：     {_pct(m.get('revenue_growth_yoy'))}",
            f"  毛利率：        {_pct(m.get('gross_margin'))}",
            f"  净利率：        {_pct(m.get('net_margin'))}",
            f"  PE（TTM）：     {m.get('pe_ratio','N/A')}",
            f"  远期 PE：       {m.get('forward_pe','N/A')}",
            f"  负债权益比：     {m.get('debt_to_equity','N/A')}",
            f"  自由现金流：     {_usd_b(m.get('free_cash_flow_b'))}",
            f"  52 周高价：     ${m.get('price_52w_high') or 0:.2f}",
            f"  52 周低价：     ${m.get('price_52w_low') or 0:.2f}",
        ]

    chain = rec.get("reasoning_chain", [])
    if chain:
        lines += ["", "多跳推理链"]
        for s in chain:
            lines += [
                f"\n  Step {s.get('step','?')}：{s.get('title','')}",
                f"    {s.get('finding','')}",
            ]
            if s.get("implication"):
                lines.append(f"    → {s.get('implication','')}")

    sources = rec.get("sources", [])
    if sources:
        lines += ["", "数据来源"]
        for src in sources:
            lines.append(f"  [{src.get('source_type','').upper()}] {src.get('name','')} — {src.get('relevance','')}")

    lines += [
        "", SEP,
        "免责声明：本报告由 AI 生成，仅供研究参考，不构成投资建议。",
        "投资决策前请咨询专业金融顾问。",
        SEP,
    ]
    return "\n".join(lines)
