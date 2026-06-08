"""
Financial Analyst 节点：5步多跳推理（异步并行版）

Step 1/2/3 并行执行，Step 4 汇总 1+2+3，Step 5 跨数据源风险综合
同时整合技术分析信号和同行业对比数据
"""
import asyncio
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.state import StockAnalysisState
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# LLM 实例在模块级别创建，避免每次调用重复初始化
_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    temperature=0.1,
)

ANALYST_SYS = (
    "你是顶级卖方研究分析师，专注基本面分析。"
    "请用中文回答，每个分析模块用3~5句话，简洁有力，附上数据依据。"
)


async def _llm_async(system: str, user: str) -> tuple[str, dict]:
    """返回 (content, usage_metadata)"""
    try:
        resp = await _llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        usage = resp.usage_metadata or {}
        return resp.content or "", usage
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return f"分析过程中出现错误: {str(e)}", {}


async def analyst_node(state: StockAnalysisState) -> dict:
    ticker      = state.get("ticker", "UNKNOWN")
    company     = state.get("company_name", ticker)
    metrics     = state.get("key_metrics") or {}
    price_data  = state.get("price_data") or {}
    financials  = state.get("financial_statements") or {}
    news        = state.get("recent_news") or []
    rag         = state.get("rag_contexts") or []
    tech        = state.get("technical_analysis") or {}
    peers       = state.get("peer_comparison") or {}
    horizon     = state.get("horizon", "medium")
    events      = list(state.get("events") or [])

    events.append({
        "event_type": "agent_start",
        "node": "analyst",
        "message": f"启动多跳推理分析：{company}（{ticker}）",
        "data": {"steps": 5, "parallel": "1+2+3"},
    })

    quarterly_rev = {}
    try:
        quarterly_rev = financials.get("quarterly_income", {}).get("Total Revenue", {})
    except Exception:
        pass

    tech_summary = tech.get("signal_summary", "技术数据不可用")
    peer_summary = peers.get("summary", "同行数据不可用")

    # ── Step 1/2/3 并行 ──────────────────────────────────────────
    s1_prompt = f"""分析 {ticker} 的营收状况：
营收增长率（YoY）：{metrics.get('revenue_growth', 'N/A')}
TTM 营收：${metrics.get('revenue_ttm', 'N/A')}
季度营收数据：{json.dumps(quarterly_rev, default=str)[:300]}
技术信号参考：{tech_summary[:100]}

请分析：增长趋势是否加速/减速、增长质量、在行业中的竞争地位。"""

    s2_prompt = f"""分析 {ticker} 的盈利质量：
毛利率：{metrics.get('gross_margin', 'N/A')}
营业利润率：{metrics.get('operating_margin', 'N/A')}
净利率：{metrics.get('net_margin', 'N/A')}
ROE：{metrics.get('roe', 'N/A')}
同行对比：{peer_summary[:100]}

重点说明利润率与营收增长的匹配程度，以及经营杠杆效应。"""

    s3_prompt = f"""分析 {ticker} 的财务健康度：
负债权益比（D/E）：{metrics.get('debt_to_equity', 'N/A')}
流动比率：{metrics.get('current_ratio', 'N/A')}
自由现金流：${metrics.get('free_cash_flow', 'N/A')}

判断财务结构是否能支撑成长故事，以及在经济下行时的抗风险能力。"""

    (s1, u1), (s2, u2), (s3, u3) = await asyncio.gather(
        _llm_async(ANALYST_SYS, s1_prompt),
        _llm_async(ANALYST_SYS, s2_prompt),
        _llm_async(ANALYST_SYS, s3_prompt),
    )

    for step_num, title, result in [
        (1, "营收分析", s1), (2, "利润率分析", s2), (3, "财务健康度", s3)
    ]:
        events.append({
            "event_type": "reasoning_step",
            "node": "analyst",
            "message": f"Step {step_num} 完成：{title}",
            "data": {"step": step_num, "title": title, "finding": result[:150]},
        })

    # ── Step 4：估值分析（汇聚 1+2+3 + 同行对比）──────────────
    horizon_note = {
        "short": "（短线关注技术面和近期催化剂）",
        "long":  "（长线关注 DCF 内在价值和竞争护城河）",
    }.get(horizon, "（中线平衡技术面与基本面）")

    s4, u4 = await _llm_async(ANALYST_SYS, f"""综合前三步分析，对 {ticker} 进行估值判断{horizon_note}：
营收：{s1[:100]}
利润：{s2[:100]}
财务健康：{s3[:100]}

估值指标：
PE（TTM）：{metrics.get('pe_ratio','N/A')} | 远期PE：{metrics.get('forward_pe','N/A')}
PB：{metrics.get('pb_ratio','N/A')} | EV/EBITDA：{metrics.get('ev_ebitda','N/A')}
当前价：${price_data.get('current_price','N/A')} | 分析师目标价：${metrics.get('analyst_target_price','N/A')}
同行对比：{peer_summary}
技术信号：{tech_summary[:120]}

判断当前估值是否合理，并给出目标价合理区间。""")

    events.append({
        "event_type": "reasoning_step",
        "node": "analyst",
        "message": "Step 4 完成：估值合理性分析",
        "data": {"step": 4, "title": "估值分析", "finding": s4[:150]},
    })

    # ── Step 5：跨数据源风险综合 ──────────────────────────────
    news_text = "\n".join(
        f"- [{n.get('published','')}] {n.get('title','')}" for n in news[:6]
    )
    rag_text = "\n---\n".join(rag[:2]) if rag else "暂无额外文件内容"

    s5, u5 = await _llm_async(ANALYST_SYS, f"""对 {ticker} 进行跨数据源风险综合：
前四步摘要：
营收={s1[:80]}...
利润={s2[:60]}...
估值={s4[:80]}...
技术信号：{tech.get('overall_signal','N/A')} — {tech.get('rsi_signal','N/A')}，{tech.get('macd_trend','N/A')}

最新新闻：
{news_text[:400]}

SEC/RAG 内容：
{rag_text[:400]}

请识别3~5个关键风险，每个风险需注明数据依据，重点关注跨数据源的信号叠加。""")

    events.append({
        "event_type": "reasoning_step",
        "node": "analyst",
        "message": "Step 5 完成：跨数据源风险综合",
        "data": {"step": 5, "title": "风险综合", "finding": s5[:150]},
    })

    # Token 汇总
    _in  = sum(u.get("input_tokens",  0) for u in [u1, u2, u3, u4, u5])
    _out = sum(u.get("output_tokens", 0) for u in [u1, u2, u3, u4, u5])
    events.append({
        "event_type": "token_usage",
        "node": "analyst",
        "message": f"Tokens — 输入 {_in:,} / 输出 {_out:,} / 合计 {_in + _out:,}",
        "data": {"input_tokens": _in, "output_tokens": _out, "total_tokens": _in + _out},
    })

    # 置信度计算
    data_quality = sum([
        1.0 if metrics    else 0.0,
        1.0 if price_data else 0.0,
        0.8 if financials else 0.0,
        0.5 if news       else 0.0,
        0.5 if rag        else 0.0,
        0.4 if tech.get("rsi_14") is not None else 0.0,
        0.3 if peers.get("peers") else 0.0,
    ])
    confidence = min(0.95, max(0.45, data_quality / 4.5))

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
