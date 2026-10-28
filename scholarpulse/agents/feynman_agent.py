"""费曼讲解智能体：学生用自己的话讲概念，AI 找漏洞。"""
from ..llm import get_llm

SYS = "你是费曼教练。判断学生能否用大白话讲清概念。输出JSON:{\"clear\":true,\"gaps\":[...],\"fix\":\"...\"}"

def check(concept: str, explain: str) -> dict:
    try:
        return get_llm().chat_json(SYS, f"概念:{concept}\n学生讲解:{explain or '(空)'}")
    except Exception:
        return {"clear": False, "gaps": ["未提供讲解"], "fix": "用一句话向同学解释"}
