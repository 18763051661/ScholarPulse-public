"""间隔复习智能体：基于 SM-2 推荐今天该复习的概念。"""
from ..llm import get_llm

SYS = "你是教务助手。根据掌握度列表给出今日复习重点。输出JSON:{\"due\":[...],\"tip\":\"...\"}"

def recommend(mastery_brief: str) -> dict:
    try:
        return get_llm().chat_json(SYS, f"掌握度:{mastery_brief}")
    except Exception:
        return {"due": [], "tip": "按SM-2到点复习即可"}
