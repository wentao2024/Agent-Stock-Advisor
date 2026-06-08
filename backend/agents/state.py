from typing import Optional
from typing_extensions import TypedDict


class StockAnalysisState(TypedDict, total=False):
    """
    LangGraph 共享状态
    total=False → 所有字段均为可选（防止 KeyError）
    """
    # 输入
    ticker: str
    company_name: str
    include_rag: bool
    horizon: str

    # 消息（直接调用方式下保持空列表）
    messages: list

    # 数据收集结果
    price_data: dict
    financial_statements: dict
    key_metrics: dict
    recent_news: list
    rag_contexts: list

    # 技术分析（technical_analyst 节点输出）
    technical_analysis: dict

    # 同行对比（peer_benchmarker 节点输出）
    peer_comparison: dict

    # 多跳推理中间产物
    revenue_analysis: str
    margin_analysis: str
    balance_sheet_analysis: str
    valuation_analysis: str
    risk_synthesis: str

    # 流程控制
    data_fetch_complete: bool
    analysis_complete: bool
    confidence_score: float

    # 输出
    recommendation: Optional[dict]
    error: Optional[str]

    # SSE 事件日志
    events: list
