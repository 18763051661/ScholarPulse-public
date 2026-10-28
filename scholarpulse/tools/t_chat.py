"""[013] t_chat LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_chat(arg: str = "") -> dict:
    """t_chat 工具。"""
    return {"tool": "t_chat", "arg": arg}
