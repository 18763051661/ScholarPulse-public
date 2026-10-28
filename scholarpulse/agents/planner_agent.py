"""
【智能体5：个性化路径规划器】
创新点：不是随机推题，而是在知识图谱上做"弱点传染"分析——
找出 ①当前薄弱 ②且是高分考点的前置依赖 的概念，按优先级推分层任务。
纯逻辑在 mastery.weak_concepts 完成，这里用 LLM 生成给学生的自然语言学习建议。
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import Concept, Mastery

SYSTEM = """你是大学生个性化学习教练。根据学生各考点掌握度，给一段不超过120字的、可执行的下一步学习建议。
要具体：先补哪个概念、为什么、用什么方式。不要空话套话。
输出JSON: {"advice":"...","priority_order":["cid1","cid2"]}"""


def plan_advice(weak: list[Concept], mastery: dict[str, Mastery]) -> dict:
    if not weak:
        return {"advice": "当前核心考点已基本掌握，建议做一套综合卷保持手感。", "priority_order": []}
    lines = "\n".join(
        f"- [{c.cid}] {c.name}：掌握度{mastery[c.cid].p:.2f}，考试权重{c.exam_weight:.2f}，"
        f"前置={c.prerequisites}" for c in weak[:5]
    )
    try:
        data = get_llm().chat_json(SYSTEM, f"薄弱考点：\n{lines}")
        return data
    except Exception:
        return {
            "advice": f"建议优先补：{ '、'.join(c.name for c in weak[:3]) }（这些是高分考点且你掌握不足）。",
            "priority_order": [c.cid for c in weak],
        }
