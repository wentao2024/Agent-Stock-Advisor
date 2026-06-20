"""
Peer Benchmarker node — peer industry valuation comparison

Compares the target company against peers on key metrics: P/E, revenue growth, net margin.
Uses the pre-defined get_peer_comparison tool.
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
        "message": f"Fetching peer industry benchmark data: {company} ({ticker})",
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
        "summary": "No peer data available",
    }

    try:
        comparison = get_peer_comparison.invoke({"ticker": ticker})
        peers = comparison.get("peers") or []

        if not peers:
            events.append({
                "event_type": "tool_result", "node": "peer_benchmarker",
                "message": "No peer data found (does not affect main analysis)", "data": {},
            })
            return {"peer_comparison": default, "events": events}

        # Compute peer averages
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

        if own_pe is not None and peer_avg_pe is not None:
            premium = (own_pe - peer_avg_pe) / peer_avg_pe
            vs_peers["pe_vs_peers"] = round(premium, 4)
            if premium > 0.2:
                summary_parts.append(f"P/E premium {premium:.0%} above peer avg ({peer_avg_pe:.1f}x)")
            elif premium < -0.2:
                summary_parts.append(f"P/E discount {abs(premium):.0%} below peer avg ({peer_avg_pe:.1f}x)")
            else:
                summary_parts.append(f"P/E in line with peers ({own_pe:.1f}x vs avg {peer_avg_pe:.1f}x)")

        if own_growth is not None and peer_avg_growth is not None:
            vs_peers["growth_vs_peers"] = round(own_growth - peer_avg_growth, 4)
            label = "above" if own_growth > peer_avg_growth else "below"
            summary_parts.append(f"Revenue growth {label} peers ({own_growth:.1%} vs {peer_avg_growth:.1%})")

        if own_margin is not None and peer_avg_margin is not None:
            vs_peers["margin_vs_peers"] = round(own_margin - peer_avg_margin, 4)
            label = "industry leading" if own_margin > peer_avg_margin else "below industry avg"
            summary_parts.append(f"Net margin {label} ({own_margin:.1%} vs {peer_avg_margin:.1%})")

        result = {
            "sector": comparison.get("sector", ""),
            "industry": comparison.get("industry", ""),
            "peers": peers,
            "peer_avg_pe": peer_avg_pe,
            "peer_avg_revenue_growth": peer_avg_growth,
            "peer_avg_net_margin": peer_avg_margin,
            "vs_peers": vs_peers,
            "summary": "; ".join(summary_parts) if summary_parts else "No valid comparison data",
        }

        peer_names = ", ".join(p.get("ticker", "") for p in peers[:3])
        events.append({
            "event_type": "reasoning_step",
            "node": "peer_benchmarker",
            "message": f"Peer comparison complete: {peer_names}",
            "data": {
                "step": 0, "title": "Peer Comparison",
                "finding": result["summary"],
            },
        })

        return {"peer_comparison": result, "events": events}

    except Exception as e:
        logger.error(f"peer_benchmarker failed: {e}")
        events.append({
            "event_type": "tool_result", "node": "peer_benchmarker",
            "message": "Peer comparison failed (does not affect main analysis)", "data": {},
        })
        return {"peer_comparison": default, "events": events}
