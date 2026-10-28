"""[007] t_review LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_review(arg: str = "") -> dict:
    """t_review 工具。"""
    return {"tool": "t_review", "arg": arg}
