"""
【核心创新】多智能体辩论评审 (Multi-Agent Debate Grading)
三个智能体公开辩论：
  - 宽松派(lenient)：倾向给分，找学生答案里的亮点
  - 严格派(strict)：倾向扣分，找逻辑漏洞和概念错误
  - 裁判(judge)：听双方辩论后给出最终分、错因分类和修复建议
"""
from __future__ import annotations
from ..llm import get_llm

LENIENT = "你是宽松派阅卷老师。学生答案里只要沾边就给分，指出值得肯定的点。输出JSON:{\"score\":0-10,\"points\":[...],\"say\":\"...\"}"
STRICT = "你是严格派阅卷老师。专挑逻辑漏洞、概念错误、遗漏要点。输出JSON:{\"score\":0-10,\"errors\":[...],\"say\":\"...\"}"
JUDGE = "你是裁判。听取宽松派和严格派意见后，给出最终分数、错因分类(概念混淆/前置缺失/审题偏差/表达不清)和一句话修复建议。输出JSON:{\"final_score\":0-10,\"cause\":\"...\",\"fix\":\"...\"}"


def debate_grade(question: str, student_answer: str, rubric: str = "") -> dict:
    g = get_llm()
    try:
        lenient = g.chat_json(LENIENT, f"题:{question}\n学生答案:{student_answer}\n{rubric}")
        strict = g.chat_json(STRICT, f"题:{question}\n学生答案:{student_answer}\n{rubric}")
        judge = g.chat_json(JUDGE, f"题:{question}\n宽松派:{lenient}\n严格派:{strict}")
        return {"debate": True, "lenient": lenient, "strict": strict, "judge": judge}
    except Exception as e:
        return {"debate": False, "error": str(e)}
