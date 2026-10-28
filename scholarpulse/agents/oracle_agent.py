"""考点押题预言智能体：预测本章最可能出大题的考点。"""
from ..llm import get_llm

SYS = "你是十年命题经验老师。根据考点权重信号，预测最可能出大题的3个考点。输出JSON:{\"predicted\":[...],\"reason\":\"...\"}"

def predict(concepts_brief: str) -> dict:
    try:
        return get_llm().chat_json(SYS, f"考点:{concepts_brief}")
    except Exception:
        return {"predicted": [], "reason": "llm_unavailable"}
