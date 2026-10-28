"""
【智能体3：分层出题器】
根据学生当前层级，为指定薄弱概念生成对应难度的简答题。
foundation层：定义/理解；consolidation：应用/比较；extension：综合/设计。
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import Concept, QuizItem

SYSTEM = """你是大学课程命题老师。按指定难度层出简答题。
- tier=foundation：考定义、原理复述
- tier=consolidation：考应用、异同比较
- tier=extension：考综合分析、设计、场景判断
输出JSON: {"items":[{"question":"题干","reference_answer":"参考答案要点"}]}
题目必须能从给定讲义内容推出，不要超纲。每题独立成题，答案要点2-4句。"""


def make_quiz(concepts: list[Concept], student_tier: str, limit: int) -> list[QuizItem]:
    if not concepts:
        return []
    lines = "\n".join(f"- [{c.cid}] {c.name}（{c.chapter}）：{c.summary}" for c in concepts[:limit])
    user = f"学生层级：{student_tier}\n请为以下考点出题：\n{lines}"
    data = get_llm().chat_json(SYSTEM, user)
    items = []
    cmap = {c.cid: c for c in concepts}
    for i, it in enumerate(data.get("items", [])):
        cid = concepts[i].cid if i < len(concepts) else concepts[0].cid
        items.append(QuizItem(
            qid=f"q_{cid}_{i}", cid=cid, tier=student_tier,
            question=it["question"], reference_answer=it.get("reference_answer", ""),
        ))
    return items
