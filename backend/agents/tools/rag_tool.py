"""ChromaDB RAG 工具：在已索引的 SEC 文件中语义检索"""
import logging
from langchain_core.tools import tool
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_retriever = None


def set_retriever(retriever):
    global _retriever
    _retriever = retriever


@tool
def search_financial_knowledge_base(query: str, ticker: str = "") -> str:
    """
    在本地 ChromaDB 知识库中语义检索公司财务信息、SEC 风险因素、业务描述。
    query 示例：'supply chain risk factors', 'revenue growth drivers', 'competitive landscape'
    """
    if _retriever is None:
        return "知识库未初始化，请先通过 /api/index/{ticker} 接口索引公司数据。"
    try:
        search_q = f"{ticker} {query}".strip() if ticker else query
        docs = _retriever.invoke(search_q)
        if not docs:
            return f"未找到与「{query}」相关的内容。"
        results = []
        for i, doc in enumerate(docs[:settings.rag_top_k], 1):
            meta = doc.metadata
            source = f"{meta.get('ticker', '')} SEC/财务数据"
            results.append(f"[{i}] {source}\n{doc.page_content[:600]}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        logger.error(f"RAG 检索错误: {e}")
        return f"检索出错: {str(e)}"


RAG_TOOLS = [search_financial_knowledge_base]
