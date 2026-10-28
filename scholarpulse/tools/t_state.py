"""[001] t_state LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_state(arg: str = "") -> dict:
    """t_state 工具。"""
    return {"tool": "t_state", "arg": arg}
