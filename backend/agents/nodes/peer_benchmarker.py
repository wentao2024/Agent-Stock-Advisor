"""
Peer Benchmarker 节点 — 同行业估值对比

对比目标公司与同行在 PE、营收增长、净利率等关键指标上的差异
使用已定义的 get_peer_comparison 工具（之前未被调用）
"""
import logging
from typing import Optional

from agents.state import StockAnalysisState
from agents.tools.market_data import get_peer_comparison

logger = logging.getLogger(__name__)


def _f(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def peer_benchmarker_node(state: StockAnalysisState) -> dict:
    ticker      = state.get("ticker", "UNKNOWN")
    company     = state.get("company_name", ticker)
    key_metrics = state.get("key_metrics") or {}
    events      = list(state.get("events") or [])

    events.append({
        "event_type": "agent_start",
        "node": "peer_benchmarker",
        "message": f"获取同行业对标数据：{company}（{ticker}）",
        "data": {"ticker": ticker},
    })

    default = {
        "sector": key_metrics.get("sector", ""),
        "industry": key_metrics.get("industry", ""),
        "peers": [],
        "peer_avg_pe": None,
        "peer_avg_revenue_growth": None,
        "peer_avg_net_margin": None,
        "vs_peers": {},
        "summary": "无同行数据可用",
    }

    try:
        comparison = get_peer_comparison.invoke({"ticker": ticker})
        peers = comparison.get("peers") or []

        if not peers:
            events.append({
                "event_type": "tool_result", "node": "peer_benchmarker",
                "message": "未找到同行数据（不影响主体分析）", "data": {},
            })
            return {"peer_comparison": default, "events": events}

        # 计算同行均值
        valid_pe     = [p["pe"] for p in peers if p.get("pe") is not None]
        valid_growth = [p["revenue_growth"] for p in peers if p.get("revenue_growth") is not None]
        valid_margin = [p["net_margin"] for p in peers if p.get("net_margin") is not None]

        peer_avg_pe     = round(sum(valid_pe)     / len(valid_pe),     2) if valid_pe     else None
        peer_avg_growth = round(sum(valid_growth) / len(valid_growth), 4) if valid_growth else None
        peer_avg_margin = round(sum(valid_margin) / len(valid_margin), 4) if valid_margin else None

        own_pe     = _f(key_metrics.get("pe_ratio"))
        own_growth = _f(key_metrics.get("revenue_growth"))
        own_margin = _f(key_metrics.get("net_margin"))

        vs_peers      = {}
        summary_parts = []

        if own_pe and peer_avg_pe:
            premium = (own_pe - peer_avg_pe) / peer_avg_pe
            vs_peers["pe_vs_peers"] = round(premium, 4)
            if premium > 0.2:
                summary_parts.append(f"PE溢价{premium:.0%}高于同行均值({peer_avg_pe:.1f}x)")
            elif premium < -0.2:
                summary_parts.append(f"PE折价{abs(premium):.0%}低于同行均值({peer_avg_pe:.1f}x)")
            else:
                summary_parts.append(f"PE与同行相近（{own_pe:.1f}x vs 均值{peer_avg_pe:.1f}x）")

        if own_growth is not None and peer_avg_growth is not None:
            vs_peers["growth_vs_peers"] = round(own_growth - peer_avg_growth, 4)
            label = "高于" if own_growth > peer_avg_growth else "低于"
            summary_parts.append(f"营收增长{label}同行（{own_growth:.1%} vs {peer_avg_growth:.1%}）")

        if own_margin is not None and peer_avg_margin is not None:
            vs_peers["margin_vs_peers"] = round(own_margin - peer_avg_margin, 4)
            label = "行业领先" if own_margin > peer_avg_margin else "低于行业均值"
            summary_parts.append(f"净利率{label}（{own_margin:.1%} vs {peer_avg_margin:.1%}）")

        result = {
            "sector": comparison.get("sector", ""),
            "industry": comparison.get("industry", ""),
            "peers": peers,
            "peer_avg_pe": peer_avg_pe,
            "peer_avg_revenue_growth": peer_avg_growth,
            "peer_avg_net_margin": peer_avg_margin,
            "vs_peers": vs_peers,
            "summary": "；".join(summary_parts) if summary_parts else "无有效对比数据",
        }

        peer_names = "、".join(p.get("ticker", "") for p in peers[:3])
        events.append({
            "event_type": "reasoning_step",
            "node": "peer_benchmarker",
            "message": f"同行对比完成：{peer_names}",
            "data": {
                "step": 0, "title": "同行对比",
                "finding": result["summary"],
            },
        })

        return {"peer_comparison": result, "events": events}

    except Exception as e:
        logger.error(f"peer_benchmarker 失败: {e}")
        events.append({
            "event_type": "tool_result", "node": "peer_benchmarker",
            "message": "同行对比失败（不影响主体分析）", "data": {},
        })
        return {"peer_comparison": default, "events": events}
