"""[030] question_grader 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 question_grader 分析，返回结构化结果。"""
    return {"agent": "question_grader", "ok": True, "task": task}
