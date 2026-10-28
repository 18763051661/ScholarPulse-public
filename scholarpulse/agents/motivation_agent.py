"""劝学信智能体：为待介入学生生成可直接转发的个性化短信。"""
from ..llm import get_llm

SYS = "你是耐心的学业导师。写一条不超过80字、温暖具体的微信消息。输出JSON:{\"letter\":\"...\"}"

def write(student_name: str, weak: str) -> str:
    try:
        return get_llm().chat_json(SYS, f"学生:{student_name};最薄弱:{weak}").get("letter", "")
    except Exception:
        return f"{student_name}同学，最近{weak}这块多花20分钟，有问题随时问我。"
