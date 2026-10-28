"""
【智能体2：噪声过滤器】
创新点：自动把知识点分为 核心考点 / 了解即可 / 垃圾冗余，
垃圾知识点不进入学习路径、不生成任务，帮学生和老师省去无效复习。
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import Concept

SYSTEM = """你是大学课程内容甄别专家。判断每个考点在考试中的价值。
对每个概念输出 noise 标签：
- core: 核心考点，必考/高频，必须掌握
- optional: 了解即可，低频/扩展
- filtered: 垃圾冗余（广告、与考试无关的题外话、重复表述、编者按）
输出JSON: {"labels":[{"cid":"...","noise":"core|optional|filtered","reason":"一句话"}]}
只输出JSON。"""


def filter_noise(concepts: list[Concept]) -> list[Concept]:
    if not concepts:
        return concepts
    brief = "\n".join(f"- [{c.cid}] {c.name}：{c.summary}（章节:{c.chapter}, 权重:{c.exam_weight}）"
                      for c in concepts)
    data = get_llm().chat_json(SYSTEM, f"待甄别概念：\n{brief}")
    label_map = {l["cid"]: l.get("noise", "core") for l in data.get("labels", [])}
    for c in concepts:
        c.noise = label_map.get(c.cid, c.noise)
    return concepts
