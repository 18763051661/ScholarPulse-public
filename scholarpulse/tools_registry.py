"""把各智能体注册成标准 LangChain @tool，供 Agent/ReAct 调用。"""
from __future__ import annotations
from langchain_core.tools import tool
from .agents import (oracle_agent, feynman_agent, socratic_agent, error_agent,
                     rootcause_agent, spaced_agent, motivation_agent, flashcard_agent,
                     twin_agent)


@tool
def predict_exam_points(concepts_brief: str) -> dict:
    """预测本章最可能出大题的考点列表。"""
    return oracle_agent.predict(concepts_brief)


@tool
def feynman_check(concept: str, explain: str) -> dict:
    """费曼学习法：检查学生能否用大白话讲清概念。"""
    return feynman_agent.check(concept, explain)


@tool
def socratic_probe(question: str, student_answer: str) -> str:
    """苏格拉底式追问：学生答错时引导其自己发现矛盾。"""
    return socratic_agent.probe(question, student_answer)


@tool
def diagnose_error(question: str, reference: str, answer: str) -> dict:
    """错因分类：概念/计算/审题/记忆模糊。"""
    return error_agent.diagnose(question, reference, answer)


@tool
def trace_root_cause(chain: list) -> dict:
    """知识缺口根因追溯。"""
    return rootcause_agent.trace([str(x) for x in chain])


@tool
def recommend_review(mastery_brief: str) -> dict:
    """间隔重复：推荐今日复习重点。"""
    return spaced_agent.recommend(mastery_brief)


@tool
def write_motivation_letter(student_name: str, weak: str) -> str:
    """生成给待介入学生的劝学短信。"""
    return motivation_agent.write(student_name, weak)


@tool
def make_flashcards(concepts_brief: str) -> list:
    """把考点转成正面问题/背面答案的闪卡。"""
    return flashcard_agent.make_cards(concepts_brief)


@tool
def build_twin_profile(sid: str) -> dict:
    """构建学生数字孪生画像并进入对抗赛。"""
    p = twin_agent.build_profile(sid)
    return {"persona": p.persona, "confidence_bias": p.confidence_bias,
            "misconceptions": [m.model_dump() for m in p.misconceptions]}


@tool
def query_weak_points(student_id: str) -> str:
    """查询某学生当前最薄弱的考点列表（含掌握度）。"""
    from .agents.tool_calling_agent import query_weak_points as _q
    return _q.invoke({"student_id": student_id})


# ---------- 教师端班级级诊断工具 ----------
@tool
def class_weak_map(tid: str = "") -> dict:
    """班级×考点掌握度热力图：该班共性薄弱考点排名（tid 留空=全校口径）。"""
    from . import storage
    concepts = storage.load_concepts()
    students = storage.load_students()
    pool = {sid: s for sid, s in students.items() if not tid or s.teacher_id == tid}
    scope = tid if pool else "全校"
    if not pool:
        pool = dict(students)
    acc: dict[str, list[float]] = {}
    for sid in pool:
        m = storage.load_mastery(sid)
        for c in concepts:
            acc.setdefault(c.cid, []).append(m[c.cid].p if c.cid in m else 1.0)
    rows = sorted(((c.name, sum(acc[c.cid]) / len(acc[c.cid])) for c in concepts),
                  key=lambda x: x[1])
    return {"scope": scope, "n_students": len(pool),
            "weak_top5": [{"concept": n, "avg": round(a, 2)} for n, a in rows[:5]],
            "strong_top3": [{"concept": n, "avg": round(a, 2)} for n, a in rows[-3:]]}


@tool
def class_risk_scan(tid: str = "") -> dict:
    """掉队风险扫描：该班（或全校）平均掌握度最低的学生名单+各自最薄弱考点。"""
    from . import storage
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    students = storage.load_students()
    pool = {sid: s for sid, s in students.items() if not tid or s.teacher_id == tid}
    if not pool:
        pool = dict(students)
    rows = []
    for sid, s in pool.items():
        m = storage.load_mastery(sid)
        vals = [x.p for x in m.values()] or [1.0]
        avg = sum(vals) / len(vals)
        worst = min((x.p for x in m.values()), default=1.0)
        weak_names = [name_of.get(cid, cid) for cid, x in m.items() if x.p <= worst + 1e-9][:3]
        rows.append({"sid": sid, "name": s.name, "avg": round(avg, 2),
                     "at_risk": avg < 0.5, "weakest": weak_names})
    rows.sort(key=lambda r: r["avg"])
    return {"scope": tid or "全校", "scanned": len(rows),
            "n_at_risk": sum(r["at_risk"] for r in rows),
            "top10": rows[:10]}


@tool
def make_quiz_for_student(student_id: str, limit: int = 3) -> dict:
    """按学生层级为其最薄弱考点生成分层自测题（题干+参考答案）。"""
    from . import storage
    from .agents import quiz_agent
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(student_id)
    student = storage.load_students().get(student_id)
    tier = student.tier if student else "foundation"
    ranked = sorted(concepts, key=lambda c: mastery[c.cid].p if c.cid in mastery else 1)
    items = quiz_agent.make_quiz(ranked[:limit], tier, limit)
    return {"student": student.name if student else student_id, "tier": tier,
            "items": [{"question": q.question, "reference": q.reference_answer}
                      for q in items]}


# ---------- 学区级多智能体治理（L0-L4） ----------
@tool
def seed_society_tool(seed: int = 42) -> dict:
    """学区级 L0：合成 80 教师 + 350 学生 + 社交图的教育社会（同 seed 可复现）。"""
    from .society.population import seed_society
    return seed_society(seed)


@tool
def run_abm(days: int = 30, seed: int = 42) -> dict:
    """学区级 L1：推进 ABM 行为仿真 days 个虚拟天（拖延/遗忘/传染），批量落盘。"""
    from .society.abm import run_days
    return run_days(days=days, seed=seed)


@tool
def governance_analyze(top_k: int = 5) -> dict:
    """学区级 L2：LangGraph Send 扇出 80 教师画像 → Top-K LLM 深析 → 总监归约报告。"""
    from .graph.hierarchy import run_governance
    return run_governance(top_k=top_k)


@tool
def epidemic_dashboard() -> dict:
    """学区级 L3：误区疫情仪表盘——SIR 曲线、逐概念 R0、超级传播者 Top10。"""
    from .society.epidemic import dashboard
    return dashboard()


@tool
def policy_sandbox(policy_text: str, days: int = 30) -> dict:
    """学区级 L4：反事实政策沙盘——同 seed 双世界 RCT，返回效应量/95%CI/分层分解/顾问叙事。"""
    from .graph.policy_sandbox import run_sandbox
    return run_sandbox(policy_text, days=days)


@tool
def policy_committee() -> dict:
    """学区级 L4b：政策听证委员会——效率/公平/风险三派两轮交叉辩论 + 主持人裁决（需先跑政策沙盘）。"""
    from .graph.policy_debate import run_debate
    try:
        return run_debate()
    except ValueError as e:
        return {"error": str(e)}


ALL_TOOLS = [predict_exam_points, feynman_check, socratic_probe, diagnose_error,
             trace_root_cause, recommend_review, write_motivation_letter, make_flashcards,
             build_twin_profile, query_weak_points,
             class_weak_map, class_risk_scan, make_quiz_for_student,
             seed_society_tool, run_abm, governance_analyze, epidemic_dashboard,
             policy_sandbox, policy_committee]
