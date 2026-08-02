from typing import Optional
from typing_extensions import TypedDict



class StockAnalysisState(TypedDict, total=False):
    """
    LangGraph shared state.
    total=False → all fields are optional (prevents KeyError)
    """
    # Inputs
    ticker: str
    company_name: str
    horizon: str

    # Messages (kept as empty list when using direct invocation)
    messages: list

    # Data collection results
    price_data: dict
    financial_statements: dict
    key_metrics: dict
    recent_news: list

    # Technical analysis (output from technical_analyst node)
    technical_analysis: dict

    # Peer comparison (output from peer_benchmarker node)
    peer_comparison: dict

    # Multi-hop reasoning intermediate outputs
    revenue_analysis: str
    margin_analysis: str
    balance_sheet_analysis: str
    valuation_analysis: str
    risk_synthesis: str

    # Flow control
    data_fetch_complete: bool
    analysis_complete: bool
    confidence_score: float

    # Output
    recommendation: Optional[dict]
    error: Optional[str]

    # Reflection loop (self-critique)
    reflection_count: int
    reflection_critique: str

    # SSE event log
    events: list
