"""
【智能体4：反幻觉校验器】（本系统核心创新）
出题智能体很容易"编造"讲义里根本没有的考点。
本节点对每道题做grounding校验：这道题/答案能否真的从上传讲义推出？
不能 -> 丢弃该题（grounded=false），避免学生复习到不存在的内容。
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import QuizItem, Concept

SYSTEM = """你是严格的命题审题人。判断每道题是否真的能从给定讲义内容推出。
对每题输出：
- grounded: true=讲义里确实有依据；false=讲义没有/超纲/编造
- source_quote: 讲义中支撑该题的原文短句（grounded=false 时留空）
输出JSON: {"results":[{"qid":"...","grounded":true,"source_quote":"原文短句"}]}
只输出JSON。宁缺毋滥，讲义中找不到依据就判false。"""


def verify_quiz(items: list[QuizItem], concepts: list[Concept], material_text: str) -> list[QuizItem]:
    if not items:
        return items
    cmap = {c.cid: c for c in concepts}
    lines = []
    for it in items:
        c = cmap.get(it.cid)
        cdesc = f"{c.name}：{c.summary}" if c else it.cid
        lines.append(f"- qid={it.qid}（考点:{cdesc}）题目:{it.question} 答案:{it.reference_answer}")
    user = f"讲义原文（节选）：\n{material_text[:2500]}\n\n待校验题目：\n" + "\n".join(lines)
    try:
        data = get_llm().chat_json(SYSTEM, user)
    except Exception:
        # 校验服务不可用时，保守起见全部标记为待人工确认（保留但不通过）
        for it in items:
            it.grounded = False
        return items
    res_map = {r["qid"]: r for r in data.get("results", [])}
    for it in items:
        r = res_map.get(it.qid, {})
        it.grounded = bool(r.get("grounded", False))
        it.source_quote = r.get("source_quote", "") or ""
    return items
