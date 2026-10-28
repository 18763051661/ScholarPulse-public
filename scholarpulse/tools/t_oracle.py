"""[010] t_oracle LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_oracle(arg: str = "") -> dict:
    """t_oracle 工具。"""
    return {"tool": "t_oracle", "arg": arg}
