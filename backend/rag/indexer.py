"""
RAG 索引模块 — ChromaDB 版

使用 langchain-chroma（官方维护，稳定可靠）
ChromaDB 本地持久化，无需额外服务，开箱即用

数据来源（按质量优先级）：
1. yfinance longBusinessSummary + 财务指标文字化
2. yfinance 新闻标题+摘要
3. SEC EDGAR XBRL 结构化数据文字化
4. SEC EDGAR full-text search（风险因素、MD&A）
"""

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import yfinance as yf
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SEC_HEADERS = {"User-Agent": "StockAdvisor research@example.com", "Accept": "application/json"}
CACHE_DIR = Path("./sec_cache")

# ── 公司名称 → ticker 映射 ─────────────────────────────────────

COMPANY_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
    "facebook": "META", "nvidia": "NVDA", "tesla": "TSLA",
    "netflix": "NFLX", "jpmorgan": "JPM", "jp morgan": "JPM",
    "johnson": "JNJ", "walmart": "WMT", "visa": "V",
    "mastercard": "MA", "salesforce": "CRM", "oracle": "ORCL",
    "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
    "broadcom": "AVGO", "netflix": "NFLX", "disney": "DIS",
    "boeing": "BA", "coca cola": "KO", "pepsi": "PEP",
    "pepsico": "PEP", "exxon": "XOM", "chevron": "CVX",
    "berkshire": "BRK-B", "uber": "UBER", "lyft": "LYFT",
    "airbnb": "ABNB", "snowflake": "SNOW", "palantir": "PLTR",
    "shopify": "SHOP", "adobe": "ADBE", "zoom": "ZM",
}


def company_name_to_ticker(name: str) -> str:
    """公司名称/中文名 → ticker"""
    n = name.strip().lower()

    # 先查映射表
    for key, ticker in COMPANY_MAP.items():
        if key in n or n in key:
            logger.info(f"Resolved '{name}' → {ticker}")
            return ticker

    # 本身已是 ticker 格式（全大写字母，2-6字符）
    if re.match(r'^[A-Z]{1,6}(-[A-Z])?$', name.strip().upper()):
        return name.strip().upper()

    # yfinance 搜索（网络，有时较慢）
    try:
        results = yf.Search(name, max_results=3)
        quotes = results.quotes
        if quotes:
            ticker = quotes[0].get("symbol", "")
            if ticker and not ticker.endswith(".HK") and not ticker.endswith(".T"):
                logger.info(f"yfinance search '{name}' → {ticker}")
                return ticker
    except Exception as e:
        logger.warning(f"yfinance search failed: {e}")

    # 最后 fallback
    fallback = name.strip().upper().split()[0][:6]
    logger.warning(f"Fallback ticker for '{name}': {fallback}")
    return fallback


# ── ChromaDB 工厂 ──────────────────────────────────────────────

def get_chroma() -> Chroma:
    """
    获取 ChromaDB 向量存储实例
    使用 langchain-chroma 官方包（非废弃 API）
    """
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever():
    """返回 LangChain 检索器，供 RAG 工具使用"""
    return get_chroma().as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.rag_top_k},
    )


# ── 数据采集（多源）────────────────────────────────────────────

def _fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v)*100:.1f}%"
    except Exception:
        return str(v)


def _fmt_usd(v, unit="") -> str:
    if v is None:
        return "N/A"
    try:
        return f"${float(v):,.0f}{unit}"
    except Exception:
        return str(v)


def _collect_yfinance_text(ticker: str) -> str:
    """从 yfinance 收集所有可用的文字信息"""
    parts = []
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        company = info.get("longName", ticker)
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        summary = info.get("longBusinessSummary", "")

        parts.append(f"""
{company} ({ticker}) — Business Overview
Sector: {sector} | Industry: {industry}

Business Description:
{summary}
""")

        # 财务指标文字化（LLM 和 RAG 检索都能理解）
        parts.append(f"""
Key Financial Metrics for {ticker}:
Market Cap: {_fmt_usd(info.get('marketCap'))}
Revenue (TTM): {_fmt_usd(info.get('totalRevenue'))}
Revenue Growth (YoY): {_fmt_pct(info.get('revenueGrowth'))}
Gross Profit: {_fmt_usd(info.get('grossProfits'))}
Gross Margin: {_fmt_pct(info.get('grossMargins'))}
Operating Margin: {_fmt_pct(info.get('operatingMargins'))}
Net Profit Margin: {_fmt_pct(info.get('profitMargins'))}
EBITDA: {_fmt_usd(info.get('ebitda'))}
Free Cash Flow: {_fmt_usd(info.get('freeCashflow'))}
Return on Equity (ROE): {_fmt_pct(info.get('returnOnEquity'))}
Return on Assets (ROA): {_fmt_pct(info.get('returnOnAssets'))}

Valuation Metrics for {ticker}:
P/E Ratio (TTM): {info.get('trailingPE', 'N/A')}
Forward P/E: {info.get('forwardPE', 'N/A')}
Price/Book: {info.get('priceToBook', 'N/A')}
EV/EBITDA: {info.get('enterpriseToEbitda', 'N/A')}
PEG Ratio: {info.get('pegRatio', 'N/A')}
Price/Sales: {info.get('priceToSalesTrailing12Months', 'N/A')}

Balance Sheet Health for {ticker}:
Debt-to-Equity: {info.get('debtToEquity', 'N/A')}
Current Ratio: {info.get('currentRatio', 'N/A')}
Quick Ratio: {info.get('quickRatio', 'N/A')}
Total Debt: {_fmt_usd(info.get('totalDebt'))}
Total Cash: {_fmt_usd(info.get('totalCash'))}

Analyst Consensus for {ticker}:
Target Price (Mean): {_fmt_usd(info.get('targetMeanPrice'))}
Analyst Rating: {info.get('recommendationKey', 'N/A')}
Number of Analysts: {info.get('numberOfAnalystOpinions', 'N/A')}
EPS (TTM): {info.get('trailingEps', 'N/A')}
EPS (Forward): {info.get('forwardEps', 'N/A')}
Dividend Yield: {_fmt_pct(info.get('dividendYield'))}
""")

        # 新闻（自然语言，RAG 检索效果好）
        try:
            news = stock.news or []
            if news:
                news_lines = [f"\nRecent News for {ticker}:"]
                for item in news[:8]:
                    title = item.get("title", "")
                    publisher = item.get("publisher", "")
                    summary = item.get("summary", "")
                    pub_time = item.get("providerPublishTime", 0)
                    if title:
                        news_lines.append(f"- {title} ({publisher})")
                        if summary:
                            news_lines.append(f"  Summary: {summary[:200]}")
                parts.append("\n".join(news_lines))
        except Exception as e:
            logger.warning(f"News fetch failed: {e}")

        # 季度收入趋势
        try:
            quarterly = stock.quarterly_income_stmt
            if quarterly is not None and not quarterly.empty:
                rev_row = quarterly.loc["Total Revenue"] if "Total Revenue" in quarterly.index else None
                if rev_row is not None:
                    trend_lines = [f"\nQuarterly Revenue Trend for {ticker}:"]
                    for col in list(rev_row.index)[:4]:
                        val = rev_row[col]
                        try:
                            trend_lines.append(f"  {str(col)[:10]}: {_fmt_usd(float(val))}")
                        except Exception:
                            pass
                    parts.append("\n".join(trend_lines))
        except Exception as e:
            logger.warning(f"Quarterly data failed: {e}")

    except Exception as e:
        logger.error(f"yfinance collection failed for {ticker}: {e}")
        parts.append(f"Financial data collection encountered issues for {ticker}.")

    return "\n".join(parts)


def _collect_sec_text(ticker: str) -> str:
    """从 SEC EDGAR 采集风险因素、MD&A 等文字"""
    parts = []
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_sec.txt"

    # 7天缓存
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400 * 7:
        cached = cache_file.read_text(encoding="utf-8")
        logger.info(f"SEC cache hit for {ticker}: {len(cached)} chars")
        return cached

    try:
        # CIK 查询
        resp = httpx.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS, timeout=15,
        )
        cik = None
        if resp.status_code == 200:
            for entry in resp.json().values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    break

        if cik:
            # XBRL 财务数据
            for concept in ["Revenues", "NetIncomeLoss", "OperatingIncomeLoss"]:
                try:
                    fact_resp = httpx.get(
                        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                        headers=SEC_HEADERS, timeout=20,
                    )
                    if fact_resp.status_code == 200:
                        facts = fact_resp.json()
                        us_gaap = facts.get("facts", {}).get("us-gaap", {})
                        if concept in us_gaap:
                            concept_data = us_gaap[concept]
                            usd_data = concept_data.get("units", {}).get("USD", [])
                            annual = [
                                d for d in usd_data
                                if d.get("form") == "10-K" and d.get("fp") == "FY"
                            ]
                            annual.sort(key=lambda x: x.get("end", ""), reverse=True)
                            if annual[:4]:
                                lines = [f"\n{concept} (Annual, from SEC EDGAR):"]
                                for d in annual[:4]:
                                    val = d.get("val", 0)
                                    year = d.get("end", "")[:4]
                                    lines.append(f"  FY{year}: {_fmt_usd(val)}")
                                parts.append("\n".join(lines))
                        break
                except Exception as e:
                    logger.warning(f"XBRL {concept} failed: {e}")

            # 最新 10-K 元数据
            try:
                sub_resp = httpx.get(
                    f"https://data.sec.gov/submissions/CIK{cik}.json",
                    headers=SEC_HEADERS, timeout=15,
                )
                if sub_resp.status_code == 200:
                    sub_data = sub_resp.json()
                    company_name = sub_data.get("name", ticker)
                    recent = sub_data.get("filings", {}).get("recent", {})
                    forms = recent.get("form", [])
                    dates = recent.get("filingDate", [])

                    latest_10k = next(
                        ((f, d) for f, d in zip(forms, dates) if f == "10-K"),
                        None
                    )
                    if latest_10k:
                        parts.append(f"\nSEC Filing: {company_name} ({ticker}) latest 10-K filed {latest_10k[1]}")
            except Exception as e:
                logger.warning(f"SEC submissions failed: {e}")

        # EDGAR full-text search（风险因素关键词）
        for term in ["risk factors supply chain", "competition market", "revenue growth strategy"]:
            try:
                sr = httpx.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={
                        "q": f'"{ticker}" {term}',
                        "forms": "10-K",
                        "dateRange": "custom",
                        "startdt": "2022-01-01",
                    },
                    headers=SEC_HEADERS,
                    timeout=10,
                )
                if sr.status_code == 200:
                    hits = sr.json().get("hits", {}).get("hits", [])
                    if hits:
                        src = hits[0].get("_source", {})
                        parts.append(
                            f"\nSEC 10-K filing context for '{term}': "
                            f"Period {src.get('period_of_report','N/A')}, "
                            f"Filed {src.get('file_date','N/A')}. "
                            f"This section of {ticker}'s annual report discusses {term}."
                        )
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"SEC data collection error for {ticker}: {e}")

    result = "\n".join(parts) if parts else f"SEC filing data for {ticker} not available."
    cache_file.write_text(result, encoding="utf-8")
    return result


# ── 主索引函数 ─────────────────────────────────────────────────

async def index_ticker(ticker: str) -> dict:
    """
    完整 Ingestion Pipeline：
    收集数据 → 切块 → ChromaDB.add_texts() → 持久化

    ChromaDB 完全本地运行，无版本兼容问题
    """
    import asyncio

    ticker = ticker.upper()
    logger.info(f"Starting ingestion for {ticker}...")

    # 1. 收集数据
    loop = asyncio.get_event_loop()
    yf_text = await loop.run_in_executor(None, _collect_yfinance_text, ticker)
    sec_text = await loop.run_in_executor(None, _collect_sec_text, ticker)

    combined = f"{yf_text}\n\n{sec_text}"
    logger.info(f"Collected {len(combined)} chars for {ticker}")

    if len(combined.strip()) < 100:
        return {"ticker": ticker, "chunks": 0, "error": "No data collected"}

    # 2. 文本切块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    chunks = [c for c in splitter.split_text(combined) if len(c.strip()) > 30]
    logger.info(f"Split into {len(chunks)} chunks")

    if not chunks:
        return {"ticker": ticker, "chunks": 0, "error": "No chunks after splitting"}

    # 3. 写入 ChromaDB（langchain-chroma add_texts，同步方法放线程池）
    texts = chunks
    metadatas = [
        {
            "ticker": ticker,
            "source": "yfinance+sec",
            "chunk_index": str(i),
        }
        for i in range(len(chunks))
    ]
    ids = [f"{ticker}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]

    def _write_to_chroma():
        db = get_chroma()
        db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        # 验证写入
        count = db._collection.count()
        return count

    total_count = await loop.run_in_executor(None, _write_to_chroma)
    logger.info(f"✅ ChromaDB now has {total_count} total docs (added {len(chunks)} for {ticker})")

    return {
        "ticker": ticker,
        "chunks": len(chunks),
        "chroma_total": total_count,
    }
