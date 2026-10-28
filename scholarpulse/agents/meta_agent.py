"""
【元认知智能体 MetaAgent】—— 本系统差异化创新
1. AI 批改简答：不只是选择/判断，老师视角给简答打分+点评
2. 信心校准(Dunning-Kruger检测)：学生答题前自报把握度，和实际得分对比，
   识别"高估自己"的概念（最危险的假性掌握）
3. 考点预言家：根据讲义信号+历年权重，预测本章最可能考的点
4. 教师劝学信：为待介入学生生成一封可直接发送的个性化短信
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import Concept, Mastery

SYS_GRADE = """你是严格的大学阅卷老师。根据参考答案评判学生作答。
输出JSON: {"score":0或0.5或1, "comment":"一句话点评(<=40字)"}。
0=完全没答到点, 0.5=部分正确, 1=基本答对。只输出JSON。"""

SYS_ORACLE = """你是有十年经验的命题老师。根据各考点在讲义中的权重信号，预测本章节最可能出大题的3个考点。
输出JSON: {"predicted":["考点名1","考点名2","考点名3"], "reason":"一句话"}"""

SYS_LETTER = """你是耐心的学业导师。给学习偏弱的学生写一条不超过80字、温暖但具体的微信消息，指出他最该补的一个点，不要空话。
直接输出JSON: {"letter":"..."}"""


def grade_answer(question: str, reference: str, student_answer: str) -> dict:
    user = f"题目:{question}\n参考答案要点:{reference}\n学生作答:{student_answer or '(空)'}"
    try:
        return get_llm().chat_json(SYS_GRADE, user)
    except Exception:
        # LLM 不可用时按关键词命中粗判
        hit = any(k in student_answer for k in reference.split("，")) if student_answer else False
        return {"score": 1.0 if hit else 0.0, "comment": "规则粗判(LLM不可用)"}


def calibrate(confidence: int, score: float) -> dict:
    """confidence 1-5, score 0/0.5/1。返回是否假性掌握。"""
    conf = confidence / 5.0
    # 高信心低得分 = 达克效应（最危险）
    over = conf >= 0.7 and score <= 0.2
    return {
        "self_confidence": conf,
        "actual": score,
        "overconfident": over,
        "note": "⚠️ 你以为会了但其实没掌握，优先重学" if over else ("把握准确，继续保持" if abs(conf - score) < 0.2 else "可再自信一点"),
    }


def exam_oracle(concepts: list[Concept]) -> dict:
    brief = "; ".join(f"{c.name}(权重{c.exam_weight})" for c in concepts if c.noise != "filtered")
    try:
        return get_llm().chat_json(SYS_ORACLE, f"考点列表:{brief}")
    except Exception:
        top = sorted(concepts, key=lambda c: -c.exam_weight)[:3]
        return {"predicted": [c.name for c in top], "reason": "按权重信号排序(LLM不可用)"}


def intervention_letter(student_name: str, weak: list[Concept]) -> str:
    if not weak:
        return ""
    user = f"学生:{student_name}; 他最薄弱的考点:{weak[0].name}; 相关:{weak[0].summary}"
    try:
        return get_llm().chat_json(SYS_LETTER, user).get("letter", "")
    except Exception:
        return f"同学你好，最近 {weak[0].name} 这块掌握不太稳，今晚花20分钟看下并做两道题吧。"
