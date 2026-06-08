"""
Stock Advisor API v3
- 输入：公司名称（支持中英文、ticker 均可）
- LLM：OpenAI gpt-4o（唯一提供商）
- 向量库：ChromaDB（本地，无需额外服务）
- 输出：SSE 流式 + 可下载 txt 报告（全中文）
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from agents.graph import compiled_graph
from agents.tools.rag_tool import set_retriever
from config import get_settings
from rag.indexer import company_name_to_ticker, get_retriever, index_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# 最近一次分析结果缓存（TTL = 10 分钟），供下载接口直接复用
_result_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 600

def _cache_put(ticker: str, rec: dict) -> None:
    _result_cache[ticker] = (time.time(), rec)

def _cache_get(ticker: str) -> Optional[dict]:
    entry = _result_cache.get(ticker)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    _result_cache.pop(ticker, None)
    return None


def _make_initial_state(ticker: str, include_rag: bool, horizon: str = "medium") -> dict:
    """显式初始化所有 state 字段，彻底防止 KeyError"""
    return {
        "ticker":                 ticker,
        "company_name":           ticker,
        "include_rag":            include_rag,
        "horizon":                horizon,
        "messages":               [],
        "price_data":             {},
        "financial_statements":   {},
        "key_metrics":            {},
        "recent_news":            [],
        "rag_contexts":           [],
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
        "recommendation":         None,
        "error":                  None,
        "events":                 [],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Stock Advisor v3（ChromaDB）启动中...")
    try:
        retriever = get_retriever()
        set_retriever(retriever)
        logger.info("✅ ChromaDB 检索器初始化完成")
    except Exception as e:
        logger.warning(f"ChromaDB 初始化警告：{e} — RAG 功能已禁用")
    yield
    logger.info("👋 服务已关闭")


app = FastAPI(
    title="Stock Advisor AI",
    description="LangGraph 多 Agent 股票分析（OpenAI + ChromaDB）",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    company_name: str
    include_rag: bool = True
    horizon: str = "medium"


# ── 健康检查 ──────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "正常",
        "model": settings.openai_model,
        "vector_db": "ChromaDB（本地）",
    }


@app.get("/api/resolve")
async def resolve(name: str = Query(...)):
    ticker = company_name_to_ticker(name)
    return {"input": name, "ticker": ticker}


# ── SEC 文件索引 ──────────────────────────────────────────────

@app.post("/api/index/{name_or_ticker}")
async def index_company(name_or_ticker: str):
    """将公司数据索引到 ChromaDB（建议分析前先运行）"""
    ticker = company_name_to_ticker(name_or_ticker)
    try:
        result = await index_ticker(ticker)
        return {"input": name_or_ticker, "ticker": ticker, **result}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── 流式分析 SSE ──────────────────────────────────────────────

@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalysisRequest):
    """SSE 流式分析，实时推送 Agent 推理过程"""
    ticker  = company_name_to_ticker(req.company_name)
    initial = _make_initial_state(ticker, req.include_rag, req.horizon)

    async def gen():
        # 每个节点返回的是"截至目前的完整 events 列表"，用 emitted 追踪已推送数量，
        # 每次只推送新增部分，避免历史事件重复出现
        emitted = 0
        try:
            async for chunk in compiled_graph.astream(
                initial,
                config={"recursion_limit": 50},
            ):
                for node_name, update in chunk.items():
                    all_events = update.get("events") or []
                    for ev in all_events[emitted:]:
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    emitted = len(all_events)
                    if update.get("recommendation"):
                        rec = update["recommendation"]
                        _cache_put(ticker, rec)
                        yield (
                            f"data: {json.dumps({'event_type':'final_recommendation','node':node_name,'message':'分析完成','data':rec}, ensure_ascii=False)}\n\n"
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
    ticker  = company_name_to_ticker(req.company_name)
    initial = _make_initial_state(ticker, req.include_rag, req.horizon)
    try:
        final = await compiled_graph.ainvoke(initial, config={"recursion_limit": 30})
        return {
            "company_name":   req.company_name,
            "ticker":         ticker,
            "recommendation": final.get("recommendation"),
            "error":          final.get("error"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── txt 报告下载 ──────────────────────────────────────────────

@app.post("/api/analyze/download")
async def analyze_download(req: AnalysisRequest):
    """返回可下载的中文 txt 报告；优先复用 10 分钟内的流式分析缓存，避免重复分析"""
    ticker = company_name_to_ticker(req.company_name)

    # 优先使用流式分析留下的缓存结果
    rec = _cache_get(ticker)

    if rec is None:
        # 缓存未命中时才重新运行分析
        initial = _make_initial_state(ticker, req.include_rag, req.horizon)
        try:
            final = await compiled_graph.ainvoke(initial, config={"recursion_limit": 50})
            rec   = final.get("recommendation")
            if not rec:
                raise HTTPException(500, f"分析失败：{final.get('error','未知错误')}")
            _cache_put(ticker, rec)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))

    txt      = _format_txt(rec, req.company_name, ticker)
    filename = f"{ticker}_分析报告_{date.today().isoformat()}.txt"
    return PlainTextResponse(
        content=txt,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/plain; charset=utf-8",
        },
    )


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
