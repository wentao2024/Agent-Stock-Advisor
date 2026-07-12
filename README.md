# Agent Stock Advisor


> AI-powered stock analysis system built with LangGraph multi-agent architecture. Input any company name, get a structured investment report with real-time streaming reasoning.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=nextdotjs)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)

## Overview

Agent Stock Advisor runs a **7-node LangGraph pipeline** for every analysis request:

```
planner → data_fetcher → technical_analyst → peer_benchmarker → analyst → synthesizer
                                                                               │
                                                             confidence < 0.65 │
                                                                    ▼          │
                                                                reflector ─────┘
                                                              (max 2 rounds)
```

Each node streams its progress to the frontend via **Server-Sent Events**, so you watch the AI reason in real time — from raw data fetching through 5-step multi-hop analysis to the final structured report.

## Features

- **Multi-hop reasoning** — 5-step analysis chain: revenue → margins → balance sheet → valuation → risk synthesis; Steps 1–3 run in parallel with `asyncio.gather`
- **Self-reflection loop** — synthesizer evaluates its own confidence (data-quality driven, not LLM subjective); low-confidence results trigger a reflector that critiques the report and re-feeds analyst
- **Technical analysis** — RSI(14), MACD(12,26,9), Bollinger Bands(20,2), MA20/50/200, volume trend — pure Python, zero LLM, fully deterministic
- **Peer benchmarking** — PE premium/discount, revenue growth, net margin vs. sector peers; peer data fetched concurrently with `ThreadPoolExecutor`
- **Real-time SSE streaming** — frontend renders agent log events as they arrive; `AbortController` for mid-stream cancellation
- **Structured output** — `with_structured_output(StockRecommendation)` enforces strict Pydantic schema on LLM output; real yfinance numbers overwrite any LLM-generated figures
- **Investment horizon** — short / medium / long; adjusts valuation focus in analyst prompts
- **Report download** — generates a formatted `.txt` report; reuses cached result within 10 min
- **Persistence** — Redis for LangGraph checkpointing (resume on reconnect); PostgreSQL for historical report storage; both optional with in-memory fallback
- **LangSmith tracing** — optional full-chain observability for all agent calls

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph 0.2 |
| LLM | OpenAI GPT-4o (`with_structured_output`) |
| API framework | FastAPI + Uvicorn (ASGI) |
| Market data | yfinance (price history, financials, news, peer metrics) |
| Technical indicators | Pure Python (no TA-Lib dependency) |
| Short-term memory | Redis + `langgraph-checkpoint-redis` |
| Long-term storage | PostgreSQL + asyncpg |
| Frontend | Next.js 14 (App Router, standalone output) |
| Containerization | Docker Compose |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/your-username/agent-stock-advisor.git
cd agent-stock-advisor

# Create root .env (read by docker-compose)
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

Optional variables (add to `.env`):

```bash
OPENAI_MODEL=gpt-4o                          # default: gpt-4o
LANGCHAIN_TRACING_V2=true                    # enable LangSmith tracing
LANGCHAIN_API_KEY=ls__your-langsmith-key
LANGCHAIN_PROJECT=stock-advisor
```

### 2. Start

```bash
docker-compose up --build
```

First run builds images — takes a few minutes. Services start in dependency order:
Redis & PostgreSQL → Backend (waits for healthcheck) → Frontend (waits for backend).

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/health |

### 3. Analyze a stock

Type any company name in the search bar — English, Chinese, or ticker symbol:

```
Apple  /  苹果  /  AAPL  /  Microsoft  /  英伟达  /  NVDA
```

Select an investment horizon (Short / Medium / Long) and click **Analyze**.

## API Reference

All endpoints accept and return JSON. The streaming endpoint uses `text/event-stream`.

### `POST /api/analyze/stream`

SSE stream. Each event is a JSON line prefixed with `data: `.

**Request:**
```json
{ "company_name": "Apple", "horizon": "medium" }
```

**Event types:**

| `event_type` | When | Key fields |
|---|---|---|
| `session_start` | Immediately | `thread_id`, `ticker` |
| `agent_start` | Node begins | `node`, `message` |
| `reasoning_step` | Mid-analysis | `step`, `title`, `finding` |
| `token_usage` | After LLM call | `input_tokens`, `output_tokens` |
| `final_recommendation` | Graph complete | `data` → full `StockRecommendation` |
| `error` | On failure | `message` |

### `POST /api/analyze`

Non-streaming (blocking). Returns full result once complete. Useful for batch processing or testing.

**Response:**
```json
{
  "company_name": "Apple",
  "ticker": "AAPL",
  "thread_id": "uuid",
  "recommendation": { ... }
}
```

### `POST /api/analyze/download`

Returns a formatted `.txt` report as a file download. Reuses the last cached result (within 10 min) to avoid redundant LLM calls.

### `GET /api/resolve?name=Apple`

Resolves a company name or Chinese name to its ticker symbol.

```json
{ "input": "Apple", "ticker": "AAPL" }
```

### `GET /api/reports/{ticker}?limit=10`

Fetches historical analysis records for a ticker (PostgreSQL mode only).

### `GET /api/health`

```json
{
  "status": "200",
  "model": "gpt-4o",
  "checkpointer": "Redis",
  "persistence": "PostgreSQL"
}
```

## Configuration

All settings are read from environment variables (`.env` in project root for Docker, `backend/.env` for local dev).

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | LLM model for all reasoning nodes |
| `REDIS_URL` | `` (disabled) | Redis URL for LangGraph checkpointing |
| `DATABASE_URL` | `` (disabled) | PostgreSQL DSN for report history |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | `` | LangSmith API key |
| `LANGCHAIN_PROJECT` | `stock-advisor` | LangSmith project name |

Redis and PostgreSQL are **optional** — the system falls back to in-memory mode when either is unavailable.

## Architecture Deep Dive

### Agent Graph

```
StockAnalysisState (TypedDict)
        │
        ├── planner            Resolves ticker; fetches company display name
        │
        ├── data_fetcher       4-way concurrent yfinance fetch (asyncio.gather)
        │                      · get_key_metrics   · get_stock_price_history
        │                      · get_financials    · get_recent_news
        │
        ├── technical_analyst  Pure-Python indicators (no LLM, no external lib)
        │                      RSI · MACD · Bollinger · MA20/50/200 · Volume trend
        │                      → Bull/Bear scoring → BULLISH / NEUTRAL / BEARISH
        │
        ├── peer_benchmarker   ThreadPoolExecutor: 3 peer tickers fetched in parallel
        │                      PE premium/discount · Revenue growth · Net margin vs sector
        │
        ├── analyst            5-step multi-hop reasoning (GPT-4o)
        │                      Steps 1+2+3 parallel via asyncio.gather
        │                      Step 4 synthesizes 1+2+3 → valuation
        │                      Step 5 cross-source risk (news + technical + valuation)
        │                      → confidence score (data-quality weighted, 0.45–0.95)
        │
        ├── synthesizer        GPT-4o with_structured_output → StockRecommendation
        │                      Overwrites LLM numbers with real yfinance data
        │
        └── reflector          (conditional) LLM quality reviewer
                               Generates 3–4 specific improvement directions
                               → re-injects into analyst system prompt
```

### Confidence Score

Confidence is computed from **data completeness**, not LLM judgment:

```python
data_quality = (
    1.0 * has_key_metrics +
    1.0 * has_price_data  +
    0.8 * has_financials  +
    0.5 * has_news        +
    0.4 * has_technical   +
    0.3 * has_peers
)
confidence = clamp(data_quality / 4.0, 0.45, 0.95)
```

Score ≥ 0.65 → route to END. Score < 0.65 → reflector (max 2 rounds).

### SSE Streaming

Each node appends to the shared `events` list in state. The stream layer tracks an `emitted` cursor and sends only the delta on each graph step. The final recommendation is sent **once**, after the graph completes — preventing duplicate DB writes during reflection loops.

## Project Structure

```
agent-stock-advisor/
├── docker-compose.yml
├── .env                          # create from .env.example
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                   # FastAPI app, lifespan, SSE/REST routes
│   ├── config.py                 # pydantic-settings
│   ├── db.py                     # asyncpg PostgreSQL CRUD
│   ├── ticker_resolver.py        # company name → ticker (map + yfinance search)
│   ├── agents/
│   │   ├── graph.py              # StateGraph definition + routing logic
│   │   ├── state.py              # StockAnalysisState TypedDict
│   │   └── nodes/
│   │       ├── analyst.py        # 5-step multi-hop reasoning
│   │       ├── synthesizer.py    # structured output → StockRecommendation
│   │       ├── reflector.py      # self-critique + improvement suggestions
│   │       ├── technical_analyst.py  # RSI / MACD / BB / MA (pure Python)
│   │       ├── peer_benchmarker.py   # concurrent peer metrics comparison
│   │       └── data_fetcher.py       # parallel yfinance data fetch
│   ├── agents/tools/
│   │   └── market_data.py        # @tool wrappers: price / financials / news / peers
│   └── models/
│       └── schemas.py            # StockRecommendation Pydantic model
└── frontend/
    ├── Dockerfile
    ├── next.config.js            # output: standalone
    └── app/
        └── page.tsx              # single-page UI: search + agent log + report
```

## StockRecommendation Schema

The final output conforms to this Pydantic model:

```python
class StockRecommendation(BaseModel):
    ticker: str
    company_name: str
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    confidence: float               # 0.0 – 1.0
    one_line_thesis: str            # ≤ 30 Chinese characters
    current_price: float
    target_price_low: Optional[float]
    target_price_high: Optional[float]
    upside_pct: Optional[float]
    bull_case: list[str]            # 3–5 items, data-backed
    bear_case: list[str]
    key_risks: list[str]
    catalysts: list[str]
    metrics: FinancialMetrics       # real yfinance numbers
    reasoning_chain: list[ReasoningStep]   # 5 steps
    sources: list[DataSource]
    investment_horizon: Literal["short", "medium", "long"]
```

## Local Development (without Docker)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in OPENAI_API_KEY
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Redis and PostgreSQL are optional — omit `REDIS_URL` and `DATABASE_URL` from `.env` to run in stateless mode.

## Useful Commands

```bash
# Run in background
docker-compose up --build -d

# Follow logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop
docker-compose down

# Full reset (wipe PostgreSQL data)
docker-compose down -v
```

## Limitations

- **Data source**: yfinance is unofficial and may be rate-limited or return stale data
- **Peer coverage**: hardcoded peer lists for 14 major tickers; other tickers get no peer comparison
- **LLM hallucination**: structured output constrains format but not reasoning quality; numerical estimates (target price) are LLM-generated
- **Not investment advice**: reports are AI-generated research aids; always consult a financial professional before making investment decisions

## License

MIT
