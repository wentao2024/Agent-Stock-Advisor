"""yfinance 市场数据工具（5个 @tool）"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _sf(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return round(float(v), 4)
    except Exception:
        return None


@tool
def get_stock_price_history(ticker: str, period: str = "1y") -> dict:
    """获取股票历史价格、52周高低、近期涨跌幅。period: 1mo/3mo/6mo/1y/2y"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return {"error": f"No price data for {ticker}"}
        current = _sf(hist["Close"].iloc[-1])
        perf = {}
        for label, days in [("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)]:
            if len(hist) >= days:
                past = hist["Close"].iloc[-days]
                if past is not None and not pd.isna(past) and past != 0:
                    perf[f"pct_{label}"] = round((hist["Close"].iloc[-1] - past) / past * 100, 2)
        ohlcv = []
        for _, row in hist.tail(30).reset_index().iterrows():
            ohlcv.append({
                "date": str(row["Date"])[:10],
                "open": _sf(row["Open"]), "high": _sf(row["High"]),
                "low": _sf(row["Low"]), "close": _sf(row["Close"]),
                "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
            })
        # 52w 高低优先使用 yfinance info（含日内高低），fallback 到历史收盘价
        try:
            info = stock.info
            high_52w = _sf(info.get("fiftyTwoWeekHigh")) or _sf(hist["High"].max())
            low_52w  = _sf(info.get("fiftyTwoWeekLow"))  or _sf(hist["Low"].min())
        except Exception:
            high_52w = _sf(hist["High"].max())
            low_52w  = _sf(hist["Low"].min())
        # closes_1y 供技术分析节点使用
        closes_1y = [_sf(c) for c in hist["Close"].tolist()]
        return {
            "ticker": ticker, "current_price": current,
            "high_52w": high_52w, "low_52w": low_52w,
            "performance": perf, "ohlcv_30d": ohlcv,
            "closes_1y": closes_1y,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_financial_statements(ticker: str) -> dict:
    """获取损益表、资产负债表、现金流量表（最近4期）"""
    try:
        stock = yf.Ticker(ticker)
        def to_dict(df, n=4):
            if df is None or df.empty:
                return {}
            df = df.iloc[:, :n]
            return {
                str(idx): {str(col)[:10]: _sf(df.loc[idx, col]) for col in df.columns}
                for idx in df.index
            }
        return {
            "ticker": ticker,
            "annual_income": to_dict(stock.income_stmt),
            "balance_sheet": to_dict(stock.balance_sheet),
            "cash_flow": to_dict(stock.cashflow),
            "quarterly_income": to_dict(stock.quarterly_income_stmt),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_key_metrics(ticker: str) -> dict:
    """获取核心财务指标：PE、PB、营收增长、利润率、市值、分析师目标价等"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector"), "industry": info.get("industry"),
            "pe_ratio": _sf(info.get("trailingPE")),
            "forward_pe": _sf(info.get("forwardPE")),
            "pb_ratio": _sf(info.get("priceToBook")),
            "ps_ratio": _sf(info.get("priceToSalesTrailing12Months")),
            "ev_ebitda": _sf(info.get("enterpriseToEbitda")),
            "peg_ratio": _sf(info.get("pegRatio")),
            "market_cap": _sf(info.get("marketCap")),
            "revenue_growth": _sf(info.get("revenueGrowth")),
            "earnings_growth": _sf(info.get("earningsGrowth")),
            "revenue_ttm": _sf(info.get("totalRevenue")),
            "gross_margin": _sf(info.get("grossMargins")),
            "operating_margin": _sf(info.get("operatingMargins")),
            "net_margin": _sf(info.get("profitMargins")),
            "roe": _sf(info.get("returnOnEquity")),
            "roa": _sf(info.get("returnOnAssets")),
            "debt_to_equity": _sf(info.get("debtToEquity")),
            "current_ratio": _sf(info.get("currentRatio")),
            "free_cash_flow": _sf(info.get("freeCashflow")),
            "eps_ttm": _sf(info.get("trailingEps")),
            "eps_forward": _sf(info.get("forwardEps")),
            "dividend_yield": _sf(info.get("dividendYield")),
            "analyst_target_price": _sf(info.get("targetMeanPrice")),
            "analyst_recommendation": info.get("recommendationKey"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "current_price": _sf(info.get("currentPrice") or info.get("regularMarketPrice")),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_recent_news(ticker: str, max_items: int = 10) -> list[dict]:
    """获取公司最新新闻，用于情绪分析和风险识别"""
    try:
        news_raw = yf.Ticker(ticker).news or []
        from datetime import datetime, timezone

        results = []
        for item in news_raw[:max_items]:
            # yfinance 新版 API：数据嵌套在 content 字段内
            c = item.get("content") or item

            title     = c.get("title", "")
            summary   = c.get("summary") or c.get("description", "")
            publisher = (c.get("provider") or {}).get("displayName", "") or c.get("publisher", "")

            # 时间：新版用 ISO 字符串 pubDate，旧版用 providerPublishTime 时间戳
            pub_date = ""
            if c.get("pubDate"):
                pub_date = c["pubDate"][:10]
            elif c.get("providerPublishTime"):
                pub_date = datetime.fromtimestamp(
                    c["providerPublishTime"], tz=timezone.utc
                ).strftime("%Y-%m-%d")

            if not title:
                continue

            results.append({
                "title": title,
                "publisher": publisher,
                "published": pub_date,
                "summary": summary,
            })

        return results
    except Exception as e:
        return [{"error": str(e)}]


def _fetch_peer_info(peer: str) -> dict | None:
    try:
        pi = yf.Ticker(peer).info
        return {
            "ticker": peer,
            "name": pi.get("longName", peer)[:25],
            "pe": _sf(pi.get("trailingPE")),
            "revenue_growth": _sf(pi.get("revenueGrowth")),
            "net_margin": _sf(pi.get("profitMargins")),
        }
    except Exception:
        return None


@tool
def get_peer_comparison(ticker: str) -> dict:
    """获取同行业关键估值指标对比"""
    PEERS = {
        "AAPL": ["MSFT", "GOOGL", "META"], "MSFT": ["AAPL", "GOOGL", "AMZN"],
        "GOOGL": ["MSFT", "META", "AMZN"], "NVDA": ["AMD", "INTC", "AVGO"],
        "TSLA": ["F", "GM", "NIO"],        "META": ["SNAP", "PINS", "GOOGL"],
        "AMZN": ["WMT", "SHOP", "BABA"],   "JPM": ["BAC", "GS", "MS"],
        "NFLX": ["DIS", "PARA", "WBD"],    "AMD":  ["NVDA", "INTC", "AVGO"],
        "INTC": ["AMD", "NVDA", "QCOM"],   "V":    ["MA", "AXP", "PYPL"],
        "WMT":  ["COST", "TGT", "AMZN"],   "DIS":  ["NFLX", "PARA", "WBD"],
    }
    try:
        info = yf.Ticker(ticker).info
        peer_list = PEERS.get(ticker.upper(), [])[:3]
        peers_data = []
        if peer_list:
            with ThreadPoolExecutor(max_workers=len(peer_list)) as executor:
                futures = {executor.submit(_fetch_peer_info, p): p for p in peer_list}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        peers_data.append(result)
        return {
            "ticker": ticker,
            "sector": info.get("sector", ""), "industry": info.get("industry", ""),
            "peers": peers_data,
        }
    except Exception as e:
        return {"error": str(e), "peers": []}


MARKET_TOOLS = [
    get_stock_price_history, get_financial_statements,
    get_key_metrics, get_recent_news, get_peer_comparison,
]
