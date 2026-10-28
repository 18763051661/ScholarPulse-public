"""
知识图谱上的提分路径规划。
创新点28/29/30：前置依赖断路检测 + 在图上求"最短提分路径"。
规则：先补前置，再补高分薄弱点；把学习顺序排出来。
"""
from __future__ import annotations
from ..models import Concept, Mastery


def prerequisite_blocks(weak: list[Concept], mastery: dict[str, Mastery]) -> list[Concept]:
    """返回按拓扑序排好的学习队列：先学前置，再学目标考点。"""
    cmap = {c.cid: c for c in weak}
    ordered: list[Concept] = []
    visited: set[str] = set()

    def visit(c: Concept):
        if c.cid in visited:
            return
        visited.add(c.cid)
        for pre in c.prerequisites:
            # 前置若也在图谱里且薄弱，先排前置
            if pre in cmap:
                visit(cmap[pre])
        ordered.append(c)

    for c in weak:
        visit(c)
    return ordered


def shortest_scoreup_plan(weak: list[Concept], mastery: dict[str, Mastery]) -> list[dict]:
    """生成按优先级的提分步骤：(考点, 当前掌握度, 考试权重, 提分收益估计)。"""
    plan = prerequisite_blocks(weak, mastery)
    out = []
    for c in plan:
        p = mastery[c.cid].p
        # 提分收益 = (1-当前掌握) × 考试权重
        gain = round((1 - p) * c.exam_weight, 2)
        out.append({"concept": c.name, "mastery": round(p, 2),
                    "exam_weight": c.exam_weight, "scoreup_gain": gain})
    out.sort(key=lambda x: -x["scoreup_gain"])
    return out


def countdown_schedule(plan: list[dict], days_left: int) -> list[dict]:
    """考试倒计时倒排：把提分步骤均匀分配到剩余天数，高分高薄弱优先前置。"""
    if not plan:
        return []
    per_day = max(1, math.ceil(len(plan) / max(1, days_left)))
    schedule = []
    for i in range(0, len(plan), per_day):
        day_no = i // per_day + 1
        schedule.append({
            "day": f"D-{days_left - day_no + 1}",
            "focus": [p["concept"] for p in plan[i:i + per_day]],
        })
    return schedule


import math  # noqa: E402
