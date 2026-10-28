"""苏格拉底追问智能体：不直接给答案，而是连续追问引导学生自己想通。"""
from ..llm import get_llm

SYS = "你是苏格拉底式导师。学生答错时，用一个反问引导他自己发现矛盾。只输出一个追问。"

def probe(question: str, student_answer: str) -> str:
    try:
        return get_llm().chat(SYS, f"题:{question}\n学生答:{student_answer}")
    except Exception:
        return "再想想：你这个结论的前提是什么？"
