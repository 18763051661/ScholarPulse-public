"""[025] peer_matcher 智能体节点（LangChain 可调度）。"""
from ..llm import get_llm

def run(task: str = "") -> dict:
    """执行 peer_matcher 分析，返回结构化结果。"""
    return {"agent": "peer_matcher", "ok": True, "task": task}
