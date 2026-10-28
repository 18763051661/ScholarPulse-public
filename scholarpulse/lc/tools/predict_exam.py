"""LangChain Tool: predict_exam"""
from langchain_core.tools import tool

@tool
def predict_exam(query: str = "") -> dict:
    """predict_exam：学情助手可用工具。"""
    return {"tool": "predict_exam", "query": query, "ok": True}
