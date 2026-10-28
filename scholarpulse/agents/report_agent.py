"""
【智能体6：15天学情报告 & 班级分析器】
学生侧：纵向对比两次知识图谱快照，说明进步/停滞/考试风险。
教师侧：聚合全班，找出班级共性薄弱点、需介入学生、分层路径预测。
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import Concept, Mastery, ReportSnapshot, Student

SYSTEM_STUDENT = """你是学情分析师。对比学生两次掌握度快照，写一份简洁报告。
输出JSON: {"summary":"一句话总评","improved":["cid"],"stagnant":["cid"],"exam_risk":"低/中/高","next_focus":"下一步重点"}"""

SYSTEM_TEACHER = """你是大学班主任/授课老师的数据助手。根据全班各考点掌握度，输出班级教学建议。
输出JSON: {"class_weak_points":["考点名"],"at_risk_students":["姓名"],"tier_path":{"foundation":["..."],"consolidation":["..."],"extension":["..."]},"teaching_advice":"一句话教学建议"}"""


def student_report(concepts: list[Concept], prev: ReportSnapshot | None,
                   cur: dict[str, float], student_name: str) -> dict:
    cmap = {c.cid: c.name for c in concepts}
    if prev is None:
        return {"summary": f"{student_name} 首次建档，已建立知识图谱。",
                "improved": [], "stagnant": [], "exam_risk": "中", "next_focus": "从核心考点开始"}
    lines = []
    for cid, p in cur.items():
        old = prev.mastery.get(cid, 0.3)
        lines.append(f"- {cmap.get(cid, cid)}: {old:.2f} -> {p:.2f}")
    try:
        data = get_llm().chat_json(SYSTEM_STUDENT, f"学生:{student_name}\n掌握度变化:\n" + "\n".join(lines))
        return data
    except Exception:
        # 降级：纯数值计算
        improved = [cid for cid in cur if prev.mastery.get(cid, 0.3) < cur[cid] - 0.1]
        weak = [cid for cid in cur if cur[cid] < 0.6]
        risk = "高" if len(weak) > len(cur) * 0.4 else ("中" if weak else "低")
        return {"summary": f"{student_name} 本周期进步{len(improved)}个考点，仍有{len(weak)}个薄弱点。",
                "improved": improved, "stagnant": weak, "exam_risk": risk,
                "next_focus": "、".join(cmap.get(c, c) for c in weak[:3])}


def teacher_dashboard(concepts: list[Concept],
                      roster: list[tuple[Student, dict[str, float]]]) -> dict:
    cmap = {c.cid: c.name for c in concepts}
    # 班级共性薄弱：统计每个考点在多少学生那里 <0.6
    weak_count: dict[str, int] = {}
    for _, m in roster:
        for cid, p in m.items():
            if p < 0.6:
                weak_count[cid] = weak_count.get(cid, 0) + 1
    sorted_weak = sorted(weak_count.items(), key=lambda x: -x[1])
    class_weak = [cmap.get(c, c) for c, _ in sorted_weak[:4]]
    # 需介入：平均掌握度最低的学生
    avg = [(s, sum(m.values()) / len(m) if m else 0) for s, m in roster]
    at_risk = [s.name for s, a in sorted(avg, key=lambda x: x[1])[:2]]
    return {"class_weak_points": class_weak, "at_risk_students": at_risk,
            "tier_path": {"foundation": [], "consolidation": [], "extension": []},
            "teaching_advice": f"全班共性薄弱点：{'、'.join(class_weak)}；建议课堂重点回顾。"}
