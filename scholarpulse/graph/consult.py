"""
【教研会诊 Multi-Teacher Consultation】——教师端的社会性创新。
与总监端"政策听证会"（价值观合议）对偶：教师端是"专业证据合议"——
一名掉队学生 → 三个教师智能体从不同职责视角独立诊断 → 教学主席归纳共识方案：
- 本班导师：班内相对位置、课堂可执行的干预
- 年级教研组长：横向对比（同考点年级表现、可复用的他班经验）
- 学习科学专家：认知机理（先修断层、遗忘曲线、误区传染）
每个角色独立调用 LLM；任一角色失败则降级为基于数据的规则视图，会诊永不空转。
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import Student

ROLES = [
    ("tutor", "本班导师",
     "聚焦该生在班内的相对位置，给出课堂内 1 周可执行的干预（坐位/提问/作业差异化）"),
    ("leader", "年级教研组长",
     "聚焦年级横向对比：该考点在年级的普遍程度，是否值得全班重讲、可否借用他班有效做法"),
    ("scientist", "学习科学专家",
     "聚焦认知机理：先修知识断层在哪、遗忘规律、同伴误区传染风险，给出认知负荷匹配的补法"),
]

SYS_ROLE = """你是中学的{title}，参加一场针对具体学生的教研会诊。
只依据给出的数据，从你的职责视角发言，120 字以内，给出 1-2 条具体可操作的做法。"""

SYS_CHAIR = """你是教学主席。综合三位教师的会诊发言，输出一段共识方案：
90 字以内，格式为「共识：…；分工：导师…，组长…，专家…；一周后复验指标：…」。"""


def _mastery_brief(sid: str) -> dict:
    """收集会诊所需的数据简报（纯数据，不调 LLM）。"""
    from .. import storage
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    students = storage.load_students()
    student = students.get(sid)
    if student is None:
        raise ValueError(f"学生 {sid} 不存在")
    m = storage.load_mastery(sid)
    ranked = sorted(concepts, key=lambda c: m[c.cid].p if c.cid in m else 1.0)
    weak = [(name_of.get(c.cid, c.cid), round(m[c.cid].p, 2)) for c in ranked[:3] if c.cid in m]
    avg = sum(x.p for x in m.values()) / len(m) if m else 1.0
    # 年级排名（全员均分）与班内排名
    def _avg(s: str) -> float:
        mm = storage.load_mastery(s)
        return sum(x.p for x in mm.values()) / len(mm) if mm else 1.0
    all_avg = sorted(((s2, _avg(s2)) for s2 in students), key=lambda x: -x[1])
    grade_rank = next(i + 1 for i, (s2, _) in enumerate(all_avg) if s2 == sid)
    mates = [s2 for s2, t in students.items() if t.teacher_id == student.teacher_id]
    mate_avg = sorted(((s2, _avg(s2)) for s2 in mates), key=lambda x: -x[1])
    class_rank = next(i + 1 for i, (s2, _) in enumerate(mate_avg) if s2 == sid)
    return {"student": student, "weak": weak, "avg": avg,
            "grade_rank": grade_rank, "grade_n": len(students),
            "class_rank": class_rank, "class_n": len(mates)}


def _brief_text(b: dict) -> str:
    weak = "、".join(f"{n}({p})" for n, p in b["weak"]) or "无"
    return (f"学生:{b['student'].name}（{b['student'].sid}，层级 {b['student'].tier}）"
            f" 班级:{b['student'].teacher_id} 班均掌握:{b['avg']:.2f}"
            f" 班内排名:{b['class_rank']}/{b['class_n']} 年级排名:{b['grade_rank']}/{b['grade_n']}"
            f" 最薄弱考点:{weak}")


def run_consult(sid: str) -> dict:
    """发起教研会诊：三角色独立诊断 → 主席共识。全降级可用。"""
    b = _mastery_brief(sid)
    brief = _brief_text(b)
    views = []
    for key, title, focus in ROLES:
        try:
            txt = get_llm().chat(SYS_ROLE.format(title=title), brief + "\n职责视角:" + focus,
                                 temperature=0.4, max_tokens=300)
        except Exception:
            n0, p0 = b["weak"][0] if b["weak"] else ("核心考点", 0.3)
            txt = (f"（规则视图）{b['student'].name} 班内 {b['class_rank']}/{b['class_n']}、"
                   f"年级 {b['grade_rank']}/{b['grade_n']}，{n0} 掌握度仅 {p0}。"
                   f"建议本周课堂优先提问该生、布置 3 道分层练习，并安排同桌结对互查。")
        views.append({"role": title, "view": txt})
    try:
        consensus = get_llm().chat(SYS_CHAIR, brief + "\n三位教师发言:\n" +
                                   "\n".join(f"[{v['role']}] {v['view']}" for v in views),
                                   temperature=0.3, max_tokens=300)
    except Exception:
        n0, p0 = b["weak"][0] if b["weak"] else ("核心考点", 0.3)
        consensus = (f"（规则共识）共识：集中补救 {n0}（当前 {p0}）。"
                     f"分工：导师每日课堂追问 1 次，组长组织同备课组共享分层题单，"
                     f"专家安排错题重做间隔为 1-2-4 天。一周后复验指标：{n0} 掌握度提升 ≥0.15。")
    return {"sid": sid, "name": b["student"].name, "brief": brief,
            "views": views, "consensus": consensus,
            "n_advisors": len(ROLES) + 1}
