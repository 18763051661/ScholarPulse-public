"""[010] planner_adaptive 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 planner_adaptive 分析，返回结构化结果。"""
    return {"agent": "planner_adaptive", "ok": True, "task": task}
