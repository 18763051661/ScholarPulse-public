"""[007] critic 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 critic 分析，返回结构化结果。"""
    return {"agent": "critic", "ok": True, "task": task}
