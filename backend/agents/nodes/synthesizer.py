"""
Synthesizer 节点：with_structured_output → StockRecommendation（中文输出）
整合基本面分析 + 技术指标 + 同行对比，生成完整结构化报告
"""
import logging
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.state import StockAnalysisState
from models.schemas import DataSource, FinancialMetrics, StockRecommendation
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SYS = """你是投资银行首席股票分析师。请基于以下分析生成完整的结构化投资报告。

要求：
- recommendation：STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
- confidence：0~1 浮点数，反映分析确定性
- one_line_thesis：核心投资论点（中文，30字以内）
- bull_case / bear_case：各3~5条，每条需有数据支撑
- key_risks：3~5个主要风险
- catalysts：近期催化剂
- reasoning_chain：5步推理链，对应分析过程
- 目标价基于 PE/DCF 合理推算，并参考分析师共识
- 所有文字内容请用中文
"""


async def synthesizer_node(state: StockAnalysisState) -> dict:
    ticker     = state.get("ticker", "UNKNOWN")
    company    = state.get("company_name", ticker)
    metrics    = state.get("key_metrics") or {}
    price_data = state.get("price_data") or {}
    news       = state.get("recent_news") or []
    rag        = state.get("rag_contexts") or []
    tech       = state.get("technical_analysis") or {}
    peers      = state.get("peer_comparison") or {}
    horizon    = state.get("horizon", "medium")
    events     = list(state.get("events") or [])

    events.append({
        "event_type": "agent_start",
        "node": "synthesizer",
        "message": "正在生成结构化投资建议报告...",
        "data": {},
    })

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    # include_raw=True 让我们能获取原始 AIMessage 以读取 usage_metadata
    structured_llm = llm.with_structured_output(StockRecommendation, include_raw=True)

    current_price = price_data.get("current_price", 0.0) or 0.0

    # 技术分析摘要
    tech_section = ""
    if tech.get("rsi_14") is not None:
        tech_section = f"""
【技术分析信号】
综合信号：{tech.get('overall_signal','N/A')}
{tech.get('signal_summary', '')}
布林带挤压：{'是（波动率收窄，注意突破方向）' if tech.get('bb_squeeze') else '否'}
股价偏离MA20：{tech.get('price_vs_ma20_pct', 'N/A')}%
"""

    # 同行对比摘要
    peer_section = ""
    if peers.get("peers"):
        peer_section = f"""
【同行业对比（{peers.get('sector','')} / {peers.get('industry','')}）】
{peers.get('summary', '')}
同行均值PE：{peers.get('peer_avg_pe','N/A')} | 均值营收增长：{peers.get('peer_avg_revenue_growth','N/A')} | 均值净利率：{peers.get('peer_avg_net_margin','N/A')}
"""

    horizon_label = {"short": "短线（<3个月）", "long": "长线（>1年）"}.get(horizon, "中线（3~12个月）")

    prompt = f"""
公司：{company}（{ticker}）| 分析日期：{date.today().isoformat()} | 当前价：${current_price} | 投资期限：{horizon_label}

【营收分析】{state.get('revenue_analysis') or '数据不足'}

【利润率分析】{state.get('margin_analysis') or '数据不足'}

【资产负债分析】{state.get('balance_sheet_analysis') or '数据不足'}

【估值分析】{state.get('valuation_analysis') or '数据不足'}

【风险综合】{state.get('risk_synthesis') or '数据不足'}

【关键财务指标】
PE（TTM）：{metrics.get('pe_ratio','N/A')} | 远期PE：{metrics.get('forward_pe','N/A')} | PB：{metrics.get('pb_ratio','N/A')}
营收增长：{metrics.get('revenue_growth','N/A')} | 净利率：{metrics.get('net_margin','N/A')}
负债率：{metrics.get('debt_to_equity','N/A')} | 自由现金流：${metrics.get('free_cash_flow','N/A')}
分析师目标价：${metrics.get('analyst_target_price','N/A')}
{tech_section}
{peer_section}
【置信度】{state.get('confidence_score', 0.6):.0%}

【最新新闻】
{chr(10).join(f'- {n.get("title","")}' for n in news[:4])}

请生成完整的结构化投资建议报告，所有文字内容使用中文。
investment_horizon 设置为：{horizon}
"""

    try:
        raw_result = await structured_llm.ainvoke([
            SystemMessage(content=SYS),
            HumanMessage(content=prompt),
        ])
        rec: StockRecommendation = raw_result["parsed"]
        if rec is None:
            raise ValueError(f"结构化输出解析失败: {raw_result.get('parsing_error')}")

        # 提取 token 用量
        raw_msg = raw_result.get("raw")
        usage   = (raw_msg.usage_metadata or {}) if raw_msg else {}
        _in  = usage.get("input_tokens",  0)
        _out = usage.get("output_tokens", 0)
        events.append({
            "event_type": "token_usage",
            "node": "synthesizer",
            "message": f"Tokens — 输入 {_in:,} / 输出 {_out:,} / 合计 {_in + _out:,}",
            "data": {"input_tokens": _in, "output_tokens": _out, "total_tokens": _in + _out},
        })

        # 用实际数据覆盖 LLM 生成的数字，确保准确性
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
            DataSource(source_type="yfinance", name=f"{ticker} 市场数据", relevance="股价、财务指标、新闻")
        ]
        if tech.get("rsi_14") is not None:
            sources.append(DataSource(
                source_type="yfinance",
                name=f"{ticker} 技术指标",
                relevance="RSI、MACD、布林带、均线系统",
            ))
        if peers.get("peers"):
            sources.append(DataSource(
                source_type="yfinance",
                name=f"同行业对比（{peers.get('sector','')}）",
                relevance="PE、营收增长、净利率行业对标",
            ))
        if rag:
            sources.append(DataSource(
                source_type="chromadb_rag",
                name=f"{ticker} SEC 文件（ChromaDB）",
                relevance="风险因素、业务描述、管理层讨论",
            ))
        if news:
            sources.append(DataSource(source_type="news", name="最新新闻", relevance="近期动态、情绪信号"))
        rec.sources = sources

        rec_dict = rec.model_dump()

        events.append({
            "event_type": "recommendation",
            "node": "synthesizer",
            "message": f"分析完成：{rec.recommendation}（置信度 {rec.confidence:.0%}）",
            "data": {
                "recommendation": rec.recommendation,
                "confidence":     rec.confidence,
                "thesis":         rec.one_line_thesis,
            },
        })
        events.append({
            "event_type": "done",
            "node": "synthesizer",
            "message": "报告生成完毕",
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
