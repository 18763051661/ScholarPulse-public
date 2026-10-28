"""闪卡智能体：把概念转成正面问题/背面答案的闪卡。"""
from ..llm import get_llm

SYS = "你是记忆法教练。为每个考点生成闪卡。输出JSON:{\"cards\":[{\"q\":\"...\",\"a\":\"...\"}]}"

def make_cards(concepts_brief: str) -> list:
    try:
        return get_llm().chat_json(SYS, f"考点:{concepts_brief}").get("cards", [])
    except Exception:
        return []
