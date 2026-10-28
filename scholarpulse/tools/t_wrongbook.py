"""[008] t_wrongbook LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_wrongbook(arg: str = "") -> dict:
    """t_wrongbook 工具。"""
    return {"tool": "t_wrongbook", "arg": arg}
