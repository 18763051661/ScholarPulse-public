"""[015] coach 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 coach 分析，返回结构化结果。"""
    return {"agent": "coach", "ok": True, "task": task}
