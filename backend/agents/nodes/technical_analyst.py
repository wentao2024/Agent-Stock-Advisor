"""
Technical Analyst node — pure quantitative indicator calculation

Computes from pre-fetched OHLCV data: RSI, MACD, Bollinger Bands, moving average system.
No LLM calls; results are fully objective and reproducible.
"""
import logging
import math
from typing import Optional

from agents.state import StockAnalysisState

logger = logging.getLogger(__name__)

def _ema(data: list, period: int) -> list:
    if not data:
        return []
    k = 2.0 / (period + 1)
    result = [data[0]]
    for v in data[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result

def _rsi(closes: list, period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

def _macd(closes: list, fast=12, slow=26, signal=9):
    if len(closes) < slow:
        return None, None, None
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    if len(macd_line) < signal:
        return None, None, None
    sig_line = _ema(macd_line, signal)
    histogram = macd_line[-1] - sig_line[-1]
    return round(macd_line[-1], 4), round(sig_line[-1], 4), round(histogram, 4)

def _bollinger(closes: list, period: int = 20, n_std: float = 2.0):
    if len(closes) < period:
        return None, None, None
    recent = closes[-period:]
    ma = sum(recent) / period
    std = math.sqrt(sum((x - ma) ** 2 for x in recent) / period)
    return round(ma + n_std * std, 4), round(ma, 4), round(ma - n_std * std, 4)


def _sma(closes: list, period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 4)


def technical_analyst_node(state: StockAnalysisState) -> dict:
    ticker     = state.get("ticker", "UNKNOWN")
    price_data = state.get("price_data") or {}
    events     = list(state.get("events") or [])

    events.append({
        "event_type": "agent_start",
        "node": "technical_analyst",
        "message": "Computing technical indicators: RSI / MACD / Bollinger Bands / Moving Averages",
        "data": {"ticker": ticker},
    })

    closes_1y: list = [c for c in (price_data.get("closes_1y") or []) if c is not None]
    ohlcv          = price_data.get("ohlcv_30d") or []
    current_price  = price_data.get("current_price") or 0.0

    result: dict = {
        "rsi_14": None, "rsi_signal": "Insufficient data",
        "macd_line": None, "macd_signal_line": None, "macd_histogram": None,
        "macd_trend": "Insufficient data",
        "bb_upper": None, "bb_middle": None, "bb_lower": None,
        "bb_position": "Insufficient data", "bb_squeeze": False,
        "ma20": None, "ma50": None, "ma200": None,
        "price_vs_ma20_pct": None, "ma_signal": "Insufficient data",
        "volume_trend": "Insufficient data",
        "overall_signal": "NEUTRAL",
        "signal_summary": "Insufficient data, cannot compute technical indicators",
    }

    if len(closes_1y) < 15:
        events.append({
            "event_type": "tool_result", "node": "technical_analyst",
            "message": "Insufficient price data, skipping technical analysis", "data": {},
        })
        return {"technical_analysis": result, "events": events}

    # ── RSI(14) ──────────────────────────────────────────────────
    rsi = _rsi(closes_1y, 14)
    result["rsi_14"] = rsi
    if rsi is not None:
        if rsi > 70:
            result["rsi_signal"] = f"Overbought ({rsi:.1f})"
        elif rsi < 30:
            result["rsi_signal"] = f"Oversold ({rsi:.1f})"
        else:
            result["rsi_signal"] = f"Neutral ({rsi:.1f})"

    # ── MACD(12,26,9) ────────────────────────────────────────────
    macd_line, macd_sig, macd_hist = _macd(closes_1y)
    result["macd_line"], result["macd_signal_line"], result["macd_histogram"] = macd_line, macd_sig, macd_hist
    if macd_hist is not None and macd_line is not None and macd_sig is not None:
        if macd_line > macd_sig and macd_hist > 0:
            result["macd_trend"] = "Bullish (MACD above signal line)"
        elif macd_line < macd_sig and macd_hist < 0:
            result["macd_trend"] = "Bearish (MACD below signal line)"
        else:
            result["macd_trend"] = "Neutral (MACD crossover zone)"

    # ── Bollinger Bands (20,2) ────────────────────────────────────
    bb_upper, bb_mid, bb_lower = _bollinger(closes_1y)
    result["bb_upper"], result["bb_middle"], result["bb_lower"] = bb_upper, bb_mid, bb_lower
    if bb_upper and bb_lower and bb_mid and current_price:
        bb_width_pct = (bb_upper - bb_lower) / bb_mid
        result["bb_squeeze"] = bb_width_pct < 0.04
        if current_price > bb_upper:
            result["bb_position"] = "Above upper band (strong)"
        elif current_price < bb_lower:
            result["bb_position"] = "Below lower band (weak)"
        elif current_price > bb_mid:
            result["bb_position"] = "Between mid and upper band (slightly bullish)"
        else:
            result["bb_position"] = "Between lower and mid band (slightly bearish)"

    # ── Moving Average System ─────────────────────────────────────
    ma20 = _sma(closes_1y, 20)
    ma50 = _sma(closes_1y, 50)
    ma200 = _sma(closes_1y, 200)
    result["ma20"], result["ma50"], result["ma200"] = ma20, ma50, ma200
    if ma20 and current_price:
        result["price_vs_ma20_pct"] = round((current_price - ma20) / ma20 * 100, 2)
    if ma20 and ma50:
        if ma20 > ma50:
            result["ma_signal"] = f"Golden cross (MA20={ma20:.2f} > MA50={ma50:.2f})"
        else:
            result["ma_signal"] = f"Death cross (MA20={ma20:.2f} < MA50={ma50:.2f})"
    elif ma20:
        result["ma_signal"] = f"MA20={ma20:.2f} (MA50 data unavailable)"

    # ── Volume Trend ──────────────────────────────────────────────
    if len(ohlcv) >= 10:
        recent_vols = [d.get("volume", 0) for d in ohlcv[-5:]]
        prev_vols   = [d.get("volume", 0) for d in ohlcv[-30:-5]]
        avg_recent = sum(v for v in recent_vols if v) / max(len(recent_vols), 1)
        avg_prev   = sum(v for v in prev_vols if v) / max(len(prev_vols), 1)
        if avg_prev > 0:
            ratio = avg_recent / avg_prev
            if ratio > 1.3:
                result["volume_trend"] = f"Expanding volume (5-day avg is {ratio:.1f}x prior period)"
            elif ratio < 0.7:
                result["volume_trend"] = f"Contracting volume (5-day avg is {ratio:.1f}x prior period)"
            else:
                result["volume_trend"] = f"Stable volume ({ratio:.1f}x)"

    # ── Bull/Bear score ───────────────────────────────────────────
    bull = 0
    bear = 0
    if rsi is not None:
        bull += (1 if rsi > 50 else 0)
        bear += (1 if rsi < 50 else 0)
    if macd_hist is not None:
        bull += (1 if macd_hist > 0 else 0)
        bear += (1 if macd_hist < 0 else 0)
    if ma20 and ma50:
        bull += (1 if ma20 > ma50 else 0)
        bear += (1 if ma20 < ma50 else 0)
    if bb_mid and current_price:
        bull += (1 if current_price > bb_mid else 0)
        bear += (1 if current_price < bb_mid else 0)
    if bull > bear + 1:
        result["overall_signal"] = "BULLISH"
    elif bear > bull + 1:
        result["overall_signal"] = "BEARISH"
    else:
        result["overall_signal"] = "NEUTRAL"

    result["signal_summary"] = (
        f"RSI: {result['rsi_signal']} | "
        f"MACD: {result['macd_trend']} | "
        f"Bollinger: {result['bb_position']} | "
        f"MA: {result['ma_signal']} | "
        f"Volume: {result['volume_trend']}"
    )

    signal_emoji = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(result["overall_signal"], "")
    events.append({
        "event_type": "reasoning_step",
        "node": "technical_analyst",
        "message": f"{signal_emoji} Technical composite signal: {result['overall_signal']}",
        "data": {
            "step": 0, "title": "Technical Analysis",
            "finding": result["signal_summary"],
        },
    })

    return {"technical_analysis": result, "events": events}
