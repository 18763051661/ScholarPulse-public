"""[018] t_badge LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_badge(arg: str = "") -> dict:
    """t_badge 工具。"""
    return {"tool": "t_badge", "arg": arg}
