"""
ScholarPulse 命令行入口（在 VSCode/IDEA 终端直接运行）：
  python run.py ingest                         # 上传示例课件，构建知识图谱
  python run.py learn --sid s001 --simulate    # 跑一轮自适应学习闭环（模拟学生答题）
  python run.py report --sid s001              # 生成 15 天学情报告 + HTML
  python run.py teacher --teacher T001         # 教师班级看板
"""
from __future__ import annotations
import sys
import random
from scholarpulse import storage, mastery as M
from scholarpulse.config import settings
from scholarpulse.models import Student, AnswerRecord, ReportSnapshot
from scholarpulse.agents import ingest_agent, filter_agent, quiz_agent, verify_agent, planner_agent, report_agent
from scholarpulse.report_html import render_student_html
from scholarpulse.core import spaced_repetition, irt, path_planner, quiz_bank, gamification, heatmap, course_manager, advanced

random.seed(42)
MATERIAL = "data/sample_lecture.txt"


def cmd_ingest():
    print("=== [1/2] 知识图谱构建智能体 ===")
    text = open(MATERIAL, encoding="utf-8").read()
    concepts = ingest_agent.build_knowledge_graph(text, MATERIAL)
    print(f"抽取到 {len(concepts)} 个考点：")
    for c in concepts:
        print(f"  - {c.name} (权重{c.exam_weight:.2f}, 前置={c.prerequisites})")
    print("\n=== [2/2] 噪声过滤智能体 ===")
    concepts = filter_agent.filter_noise(concepts)
    for c in concepts:
        print(f"  [{c.noise:>8}] {c.name}")
    # 合并入库而非整体替换——防止把全学科 900 考点清成单课件考点
    bank = {c.cid: c for c in storage.load_concepts()}
    for c in concepts:
        bank[c.cid] = c
    storage.save_concepts(list(bank.values()))
    kept = [c.name for c in concepts if c.noise != "filtered"]
    print(f"\n已过滤 {len(concepts)-len(kept)} 个垃圾/冗余知识点，保留 {len(kept)} 个进入学习路径。")


def _ensure_student(sid: str) -> Student:
    students = storage.load_students()
    if sid not in students:
        students[sid] = Student(sid=sid, name=f"学生{sid}", teacher_id="T001",
                                tier="foundation", joined_at=storage.now_iso())
        storage.save_students(students)
    return students[sid]


def cmd_learn(sid: str, simulate: bool):
    concepts = storage.load_concepts()
    if not concepts:
        print("请先运行 python run.py ingest"); return
    student = _ensure_student(sid)
    material_text = open(MATERIAL, encoding="utf-8").read()

    mastery = storage.load_mastery(sid)
    if not mastery:
        mastery = M.init_mastery(concepts)
    storage.save_mastery(sid, mastery)

    for rnd in range(1, settings.max_loop_rounds + 1):
        tier = M.student_tier(M.avg_mastery(mastery))
        student.tier = tier
        weak = M.weak_concepts(mastery, concepts)
        print(f"\n===== 第 {rnd} 轮 | 学生层级={tier} | 平均掌握={M.avg_mastery(mastery):.2f} =====")
        if not weak:
            print("🎉 所有核心考点已达标，进入综合提升。"); break
        advice = planner_agent.plan_advice(weak, mastery)
        print(f"教练建议：{advice.get('advice','')}")

        items = quiz_agent.make_quiz(weak, tier, settings.max_quiz_per_round)
        items = verify_agent.verify_quiz(items, concepts, material_text)
        kept = [i for i in items if i.grounded]
        dropped = [i for i in items if not i.grounded]
        print(f"出题 {len(items)} 道，反幻觉校验通过 {len(kept)} 道，丢弃编造题 {len(dropped)} 道：")
        for i in kept:
            print(f"  ✅ [{i.cid}] {i.question}")
        for i in dropped:
            print(f"  🚫 已丢弃（讲义无依据）：{i.question}")

        # 答题
        records = []
        for i in kept:
            if simulate:
                # 模拟学生：foundation层答对率高，advanced层低
                correct = random.random() < (0.75 if tier == "foundation" else 0.55)
            else:
                ans = input(f"\n题目：{i.question}\n你的回答(回车跳过/输y算对,其他算错): ").strip().lower()
                correct = ans == "y" or ans == "是"
            records.append(AnswerRecord(qid=i.qid, cid=i.cid, correct=correct, ts=storage.now_iso()))
            M.apply_answer(mastery[i.cid], correct)
            print(f"    -> {'[答对]' if correct else '[答错]'}（{i.cid} 掌握度 {mastery[i.cid].p:.2f}）")
        storage.save_mastery(sid, mastery)
        storage.append_answers(sid, records)

    # 保存快照用于纵向对比
    storage.append_snapshot(sid, ReportSnapshot(ts=storage.now_iso(),
                                                 mastery={c.cid: c.p for c in mastery.values()}))
    students = storage.load_students(); students[sid] = student
    storage.save_students(students)
    print("\n✅ 本轮学习闭环结束，已更新掌握度图谱。")


def cmd_report(sid: str):
    concepts = storage.load_concepts()
    student = storage.load_students()[sid]
    mastery = storage.load_mastery(sid)
    snaps = storage.load_snapshots(sid)
    prev = snaps[-2] if len(snaps) >= 2 else None
    cur = {c.cid: c.p for c in mastery.values()}
    rep = report_agent.student_report(concepts, prev, cur, student.name)
    print("=== 学情报告 ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    path = render_student_html(student, concepts, mastery, rep)
    print(f"\nHTML 报告已生成：{path}")


def cmd_teacher(teacher_id: str):
    concepts = storage.load_concepts()
    students = storage.load_students()
    roster = []
    for sid, s in students.items():
        m = storage.load_mastery(sid)
        roster.append((s, {c.cid: c.p for c in m.values()}))
    if not roster:
        print("班级暂无学生数据。"); return
    board = report_agent.teacher_dashboard(concepts, roster)
    print(f"=== 教师 {teacher_id} 班级看板 ===")
    for k, v in board.items():
        print(f"  {k}: {v}")


# ========== 扩展创新功能命令 ==========
def cmd_review(sid: str):
    """间隔重复：今天到点该复习哪些概念。"""
    mastery = storage.load_mastery(sid)
    concepts = {c.cid: c for c in storage.load_concepts()}
    due = spaced_repetition.due_review(mastery)
    print(f"=== 今日复习清单（间隔重复）===")
    if not due:
        print("暂无到点复习任务，继续学新内容即可。")
    for cid in due:
        c = concepts.get(cid)
        print(f"  🔁 {c.name if c else cid}（上次掌握 {mastery[cid].p:.2f}）")


def cmd_path(sid: str):
    """最短提分路径：图上排好学习顺序。"""
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    weak = M.weak_concepts(mastery, concepts)
    plan = path_planner.shortest_scoreup_plan(weak, mastery)
    print("=== 最短提分路径（按提分收益排序）===")
    for i, p in enumerate(plan, 1):
        print(f"  {i}. {p['concept']}  掌握{p['mastery']}  权重{p['exam_weight']}  提分收益{p['scoreup_gain']}")


def cmd_countdown(sid: str, days: int, course: str):
    """考试倒计时倒排计划。"""
    cd = course_manager.days_to_exam(course)
    days = cd if cd is not None else days
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    weak = M.weak_concepts(mastery, concepts)
    plan = path_planner.shortest_scoreup_plan(weak, mastery)
    sched = path_planner.countdown_schedule(plan, days)
    print(f"=== 距考试 {days} 天倒排计划 ===")
    for s in sched:
        print(f"  {s['day']}: {'、'.join(s['focus'])}")


def cmd_wrongbook(sid: str):
    """错题本。"""
    records = storage.load_answers(sid)
    wb = quiz_bank.wrong_book(records)
    concepts = {c.cid: c for c in storage.load_concepts()}
    print("=== 错题本 ===")
    if not wb:
        print("暂无错题，状态很好。")
    for w in wb:
        c = concepts.get(w["cid"])
        print(f"  📕 {c.name if c else w['cid']} 错{w['wrong_count']}次")


def cmd_heatmap(sid: str):
    mastery = storage.load_mastery(sid)
    concepts = storage.load_concepts()
    path = heatmap.render_heatmap(sid, concepts, mastery)
    print(f"掌握热力图已生成：{path}")


def cmd_streak(sid: str):
    records = storage.load_answers(sid)
    avg = M.avg_mastery(storage.load_mastery(sid))
    print(f"连续学习：{gamification.streak_days(records)} 天")
    print(f"获得徽章：{', '.join(gamification.achievements(records, avg))}")


def cmd_courses():
    print("=== 已选课程 ===")
    for code, c in course_manager.list_courses().items():
        d = course_manager.days_to_exam(code)
        print(f"  [{code}] {c['name']}  考试 {c['exam_date']}（剩{d}天）")


def cmd_irt(sid: str):
    records = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    diff = irt.estimate_difficulty(records)
    theta = irt.ability_from_mastery(M.avg_mastery(mastery))
    print(f"学生能力值 θ={theta:.2f}")
    print("题目难度（1=最难）：", diff or "暂无答题记录")


# ========== 高级原创智能 ==========
def cmd_advanced(sid: str):
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    records = storage.load_answers(sid)
    print("=== 概念混淆矩阵（疑似搞混的考点对）===")
    cm = advanced.confusion_matrix(records, concepts)
    print("  " + ("；".join(f"{x['pair']}({x['co_wrong']}次)" for x in cm) or "暂未发现"))
    print("\n=== 知识缺口根因追溯（以死锁为例）===")
    dl = next((c.cid for c in concepts if "死锁" in c.name), concepts[0].cid)
    chain = advanced.root_cause(dl, mastery, concepts)
    print("  " + " ← ".join(chain))
    print("\n=== 掌握度置信区间（样本少时不可尽信）===")
    for cid, m in list(mastery.items())[:4]:
        lo, hi = advanced.confidence_interval(m.p, m.n_correct + m.n_wrong)
        print(f"  {cid}: {m.p:.2f} ∈ [{lo},{hi}]")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    args = sys.argv[2:]
    sid = "s001"
    if "--sid" in args:
        sid = args[args.index("--sid") + 1]
    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "learn":
        cmd_learn(sid, "--simulate" in args)
    elif cmd == "report":
        cmd_report(sid)
    elif cmd == "teacher":
        tid = args[args.index("--teacher") + 1] if "--teacher" in args else "T001"
        cmd_teacher(tid)
    elif cmd == "review":
        cmd_review(sid)
    elif cmd == "path":
        cmd_path(sid)
    elif cmd == "countdown":
        days = int(args[args.index("--days") + 1]) if "--days" in args else 30
        course = args[args.index("--course") + 1] if "--course" in args else "OS"
        cmd_countdown(sid, days, course)
    elif cmd == "wrongbook":
        cmd_wrongbook(sid)
    elif cmd == "heatmap":
        cmd_heatmap(sid)
    elif cmd == "streak":
        cmd_streak(sid)
    elif cmd == "courses":
        cmd_courses()
    elif cmd == "irt":
        cmd_irt(sid)
    elif cmd == "graph-demo":
        from scholarpulse import langgraph_app
        print("=== 运行 LangGraph 编译状态图（plan→quiz→answer→条件回边）===")
        out = langgraph_app.run(sid)
        print(f"完成，共 {out.get('round')} 轮，剩余薄弱 {out.get('weak_count')} 个")
    elif cmd == "advanced":
        cmd_advanced(sid)
    else:
        print(__doc__)
