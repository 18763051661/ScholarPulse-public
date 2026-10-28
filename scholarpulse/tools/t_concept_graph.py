"""[003] t_concept_graph LangChain Tool 包装。"""
from langchain_core.tools import tool

@tool
def t_concept_graph(arg: str = "") -> dict:
    """t_concept_graph 工具。"""
    return {"tool": "t_concept_graph", "arg": arg}
