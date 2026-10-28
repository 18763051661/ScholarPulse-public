"""[002] t_mastery LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_mastery(arg: str = "") -> dict:
    """t_mastery 工具。"""
    return {"tool": "t_mastery", "arg": arg}
