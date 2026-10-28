"""[015] t_export LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_export(arg: str = "") -> dict:
    """t_export 工具。"""
    return {"tool": "t_export", "arg": arg}
