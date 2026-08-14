from typing import Literal, Optional
from pydantic import BaseModel, Field


class FinancialMetrics(BaseModel):
    current_price: float = 0.0
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    price_change_1y_pct: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    market_cap_b: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    revenue_ttm_b: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cash_flow_b: Optional[float] = None


class ReasoningStep(BaseModel):
    step: int
    title: str
    finding: str
    data_points: list[str] = []
    implication: str = ""


class DataSource(BaseModel):
    source_type: Literal["yfinance", "sec_edgar", "news"]
    name: str
    relevance: str


class StockRecommendation(BaseModel):
    ticker: str
    company_name: str
    analysis_date: str
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    confidence: float = Field(ge=0.0, le=1.0)
    one_line_thesis: str
    current_price: float = 0.0
    target_price_low: Optional[float] = None
    target_price_high: Optional[float] = None
    upside_pct: Optional[float] = None
    bull_case: list[str] = []
    bear_case: list[str] = []
    key_risks: list[str] = []
    catalysts: list[str] = []
    metrics: FinancialMetrics = Field(default_factory=FinancialMetrics)
    reasoning_chain: list[ReasoningStep] = []
    sources: list[DataSource] = []
    investment_horizon: Literal["short", "medium", "long"] = "medium"


class AnalysisRequest(BaseModel):
    company_name: str
    horizon: str = "medium"
