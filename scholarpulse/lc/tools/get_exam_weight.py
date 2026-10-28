"""LangChain Tool: get_exam_weight"""
from langchain_core.tools import tool

@tool
def get_exam_weight(query: str = "") -> dict:
    """get_exam_weight：学情助手可用工具。"""
    return {"tool": "get_exam_weight", "query": query, "ok": True}
