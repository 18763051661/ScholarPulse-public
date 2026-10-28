"""ScholarPulse Web 后端：学生上传课件/刷题/看报表，老师登录看班级。"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scholarpulse import storage, mastery as M
from scholarpulse.models import Student, AnswerRecord, ReportSnapshot, TwinProfile, TwinBattle
from scholarpulse.agents import (ingest_agent, filter_agent, quiz_agent, verify_agent,
                                 planner_agent, report_agent, meta_agent, twin_agent)
from scholarpulse.core import (spaced_repetition, path_planner, quiz_bank,
                               gamification)

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    from scholarpulse import scheduler
    scheduler.start_background()
    print("[scheduler] 15天报表调度已启动")
    yield


app = FastAPI(title="ScholarPulse 学脉", lifespan=lifespan)
STATIC = os.path.join(os.path.dirname(__file__), "static")


def _ensure_student(sid: str, name: str = "") -> Student:
    students = storage.load_students()
    if sid not in students:
        students[sid] = Student(sid=sid, name=name or f"学生{sid}", teacher_id="T001",
                                tier="foundation", joined_at=storage.now_iso())
        storage.save_students(students)
    return students[sid]


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/teacher")
def teacher_page():
    return FileResponse(os.path.join(STATIC, "teacher.html"))


class LoginIn(BaseModel):
    sid: str
    name: str = ""


@app.post("/api/student/login")
def student_login(body: LoginIn):
    s = _ensure_student(body.sid, body.name)
    return {"sid": s.sid, "name": s.name, "tier": s.tier, "teacher_id": s.teacher_id}


def merge_ingested(sid: str, concepts: list) -> None:
    """上传抽取的概念合并入库：同 cid 更新、新 cid 追加——上传单门课件
    不再清空全学科概念库（曾导致思维导图只剩操作系统），也保留学生其他课程掌握度。"""
    bank = {c.cid: c for c in storage.load_concepts()}
    for c in concepts:
        bank[c.cid] = c
    storage.save_concepts(list(bank.values()))
    mastery = storage.load_mastery(sid)
    mastery.update(M.init_mastery([c for c in concepts if c.cid not in mastery]))
    storage.save_mastery(sid, mastery)


@app.post("/api/upload")
async def upload(sid: str = Form(...), file: UploadFile = File(...)):
    from scholarpulse.core.lecture_parser import parse_lecture
    _ensure_student(sid)
    os.makedirs("data/uploads", exist_ok=True)
    # 按原扩展名落盘，再用统一解析器转纯文本（支持 txt/pdf/docx）
    safe_name = f"{sid}_{file.filename or 'lecture'}"
    raw_path = f"data/uploads/{safe_name}"
    with open(raw_path, "wb") as f:
        f.write(await file.read())
    content = parse_lecture(raw_path)
    open(f"data/uploads/{sid}.txt", "w", encoding="utf-8").write(content)
    concepts = ingest_agent.build_knowledge_graph(content, file.filename)
    concepts = filter_agent.filter_noise(concepts)
    merge_ingested(sid, concepts)
    kept = [c for c in concepts if c.noise != "filtered"]
    return {"extracted": len(concepts), "filtered": len(concepts) - len(kept),
            "format": (file.filename or "").split(".")[-1],
            "chars": len(content),
            "kept": [c.name for c in kept]}


def _material_of(sid: str) -> str:
    """优先读学生上传的课件，没有再退回示例。"""
    p = f"data/uploads/{sid}.txt"
    if os.path.exists(p):
        return open(p, encoding="utf-8").read()
    return open("data/sample_lecture.txt", encoding="utf-8").read()


@app.get("/api/state")
def state(sid: str):
    _ensure_student(sid)
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    if not mastery:
        mastery = M.init_mastery(concepts); storage.save_mastery(sid, mastery)
    weak = M.weak_concepts(mastery, concepts)
    return {
        "avg": round(M.avg_mastery(mastery), 2),
        "tier": M.student_tier(M.avg_mastery(mastery)),
        "concepts": [{"cid": c.cid, "name": c.name, "weight": c.exam_weight,
                      "mastery": round(mastery[c.cid].p, 2) if c.cid in mastery else None,
                      "noise": c.noise, "prereq": c.prerequisites} for c in concepts],
        "weak_plan": path_planner.shortest_scoreup_plan(weak, mastery),
        "review_due": spaced_repetition.due_review(mastery),
        "streak": gamification.streak_days(storage.load_answers(sid)),
    }


@app.get("/api/quiz/next")
def quiz_next(sid: str):
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    # 优先：自适应题库（CAT）——按学生 Elo 能力值选难度最接近、且偏薄弱考点的题
    bank = storage.load_quiz_bank()
    if bank:
        elo_d = storage.load_quiz_elo(sid)
        done = {a.qid for a in storage.load_answers(sid)}
        base = {"foundation": 1000, "consolidation": 1400, "extension": 1800}
        weak_ids = {c.cid for c in M.weak_concepts(mastery, concepts)}
        cand = [q for q in bank if q.qid not in done]
        cand.sort(key=lambda q: (q.cid not in weak_ids,
                                 abs(base.get(q.tier, 1200) - elo_d["elo"])))
        if cand:
            q = cand[0]
            json.dump({"qid": q.qid, "cid": q.cid, "question": q.question,
                       "reference": q.reference_answer, "tier": q.tier},
                      open(f"data/pending_{sid}.json", "w", encoding="utf-8"), ensure_ascii=False)
            tier_cn = {"foundation": "基础", "consolidation": "进阶", "extension": "迁移"}[q.tier]
            return {"done": False, "question": q.question, "qid": q.qid, "cid": q.cid,
                    "tier": q.tier, "tier_cn": tier_cn, "elo": elo_d["elo"],
                    "adaptive": True}
    # 回退：实时 LLM 出题（题库未生成时）
    weak = M.weak_concepts(mastery, concepts)
    if not weak:
        return {"done": True, "msg": "🎉 所有考点已达标！"}
    tier = M.student_tier(M.avg_mastery(mastery))
    items = quiz_agent.make_quiz(weak, tier, 1)
    items = verify_agent.verify_quiz(items, concepts, _material_of(sid))
    items = [i for i in items if i.grounded]
    if not items:
        return {"done": True, "msg": "本轮无可用题目，稍后再试。"}
    q = items[0]
    json.dump({"qid": q.qid, "cid": q.cid, "question": q.question, "reference": q.reference_answer},
              open(f"data/pending_{sid}.json", "w", encoding="utf-8"), ensure_ascii=False)
    return {"done": False, "question": q.question, "qid": q.qid, "cid": q.cid}


class AnswerIn(BaseModel):
    sid: str
    qid: str
    cid: str
    student_answer: str
    confidence: int


@app.post("/api/answer")
def answer(body: AnswerIn):
    pend = f"data/pending_{body.sid}.json"
    if not os.path.exists(pend):
        raise HTTPException(400, "没有待作答的题目")
    p = json.load(open(pend, encoding="utf-8"))
    grade = meta_agent.grade_answer(p["question"], p["reference"], body.student_answer)
    score = float(grade.get("score", 0)); correct = score >= 0.6
    mastery = storage.load_mastery(body.sid)
    if body.cid in mastery:
        M.apply_answer(mastery[body.cid], correct); storage.save_mastery(body.sid, mastery)
    # Elo 能力值更新（CAT 核心）：期望胜率 × K=32，答难题涨得多、答易题跌得多
    base = {"foundation": 1000, "consolidation": 1400, "extension": 1800}
    elo_d = storage.load_quiz_elo(body.sid)
    r_q = base.get(p.get("tier"), 1200)
    expect = 1.0 / (1.0 + 10 ** ((r_q - elo_d["elo"]) / 400.0))
    delta = round(32 * ((1.0 if correct else 0.0) - expect))
    elo_d["elo"] = max(800, elo_d["elo"] + delta); elo_d["n"] += 1
    storage.save_quiz_elo(body.sid, elo_d)
    storage.append_answers(body.sid, [AnswerRecord(qid=body.qid, cid=body.cid,
                          correct=correct, ts=storage.now_iso(),
                          question=p["question"], reference=p["reference"])])
    return {"score": score, "comment": grade.get("comment", ""),
            "your_mastery": round(mastery[body.cid].p, 2),
            "calibration": meta_agent.calibrate(body.confidence, score),
            "elo": elo_d["elo"], "elo_delta": delta}


# ---------- 在线考试系统：多课程选科 + AI防作弊监考 + CAT自适应 + 结构化成绩单 ----------
@app.get("/api/exam/courses")
def exam_courses():
    """列出可选课程及各课题量、已答进度。博士级创新：覆盖计算机专业核心10门课。"""
    from scholarpulse.core.curriculum import COURSES
    bank = storage.load_quiz_bank()
    concepts = storage.load_concepts()
    course_n: dict[str, int] = {}
    course_concepts: dict[str, int] = {}
    for q in bank:
        c = getattr(q, "course", "os")
        course_n[c] = course_n.get(c, 0) + 1
    for c in concepts:
        co = getattr(c, "course", "os")
        course_concepts[co] = course_concepts.get(co, 0) + 1
    courses = []
    for code, name in COURSES.items():
        n_q = course_n.get(code, 0)
        if code == "os" and n_q == 0:  # OS 用旧题库
            n_q = len([q for q in bank if getattr(q, "course", "os") == "os"])
        courses.append({"code": code, "name": name,
                        "n_quiz": n_q, "n_concepts": course_concepts.get(code, 0)})
    # 也加上按 course 字段统计但不在 COURSES 里的
    known = set(COURSES.keys())
    for code, n in course_n.items():
        if code not in known:
            courses.append({"code": code, "name": code, "n_quiz": n,
                            "n_concepts": course_concepts.get(code, 0)})
    return {"courses": courses, "total_quiz": len(bank), "total_courses": len(courses)}


@app.post("/api/exam/start")
def exam_start(sid: str, course: str = "os", mode: str = "fixed"):
    """模拟考试组卷。支持多课程选科 + CAT自适应两种模式。

    course: 课程代号（os/ds/net/db/co/algo/dm/comp/se/sec），默认 os。
    mode: 'fixed'=固定10题按权重抽（难度4基/4进/2迁），
          'cat'=CAT自适应：按学生 Elo 能力值动态配难度阶梯。
    落盘待考卷（30 分钟限时）。"""
    bank = storage.load_quiz_bank()
    if not bank:
        raise HTTPException(400, "题库未生成，请先运行 python -m scholarpulse.core.quiz_bank")
    concepts = storage.load_concepts()
    course_concepts = {c.cid: c for c in concepts
                       if getattr(c, "course", "os") == course}
    w_of = {c.cid: c.exam_weight for c in concepts}
    course_bank = [q for q in bank if getattr(q, "course", "os") == course]
    if not course_bank:
        raise HTTPException(400, f"课程 {course} 题库为空")
    done = {a.qid for a in storage.load_answers(sid)}

    if mode == "cat":
        # CAT 自适应：按 Elo 能力值确定起始难度阶梯，构建自适应卷
        elo_d = storage.load_quiz_elo(sid)
        elo = elo_d.get("elo", 1200)
        # Elo→起始 tier 映射：<1000 foundation, 1000-1400 consolidation, >1400 extension
        tier_order = ["foundation", "consolidation", "extension"]
        tier_base = {"foundation": 1000, "consolidation": 1400, "extension": 1800}
        # 找当前 Elo 对应 tier
        start_idx = 0 if elo < 1000 else 1 if elo < 1400 else 2
        paper, seen = [], set()
        # 从起始 tier 开始，逐步加入更难/更易的题，构建 10 题自适应阶梯
        for offset in [0, 0, 1, -1, 1, 0, 1, -1, 1, 0]:
            idx = max(0, min(2, start_idx + offset))
            tier = tier_order[idx]
            avail = [q for q in course_bank
                     if q.tier == tier and q.qid not in done and q.cid not in seen]
            avail.sort(key=lambda q: -w_of.get(q.cid, 0))
            if avail:
                paper.append(avail[0]); seen.add(avail[0].cid)
            if len(paper) >= 10:
                break
        exam_mode = "cat"
    else:
        # 固定模式：难度配比 4 基础/4 进阶/2 迁移，考点去重
        tiers: dict[str, list] = {"foundation": [], "consolidation": [], "extension": []}
        for q in course_bank:
            if q.qid not in done:
                tiers.get(q.tier, []).append(q)
        paper, seen = [], set()
        quota = {"foundation": 4, "consolidation": 4, "extension": 2}
        for k in ("foundation", "consolidation", "extension"):
            tiers[k].sort(key=lambda q: -w_of.get(q.cid, 0))
            for q in tiers[k]:
                if q.cid not in seen:
                    paper.append(q); seen.add(q.cid)
                if sum(1 for p in paper if p.tier == k) >= quota[k]:
                    break
        exam_mode = "fixed"
    paper = paper[:10]
    if len(paper) < 5:
        raise HTTPException(400, f"课程 {course} 未做过的题目不足，无法组卷")
    exam = {"ts": storage.now_iso(), "deadline_min": 30, "course": course,
            "mode": exam_mode, "elo_start": storage.load_quiz_elo(sid).get("elo", 1200),
            "items": [{"qid": q.qid, "cid": q.cid, "q": q.question,
                       "ref": q.reference_answer, "tier": q.tier} for q in paper]}
    json.dump(exam, open(f"data/exam_{sid}.json", "w", encoding="utf-8"), ensure_ascii=False)
    return {"total": len(exam["items"]), "deadline_min": 30, "course": course,
            "mode": exam_mode, "elo_start": exam["elo_start"],
            "items": [{"i": i, "q": it["q"], "tier": it["tier"]} for i, it in enumerate(exam["items"])]}


class ExamSubmitIn(BaseModel):
    sid: str
    answers: dict[str, str]   # 题号(i) -> 作答
    timing: dict[str, int] = {}   # 题号(i) -> 耗时(秒)，供监考官取证
    camera: dict = {}   # 摄像头视觉统计（前端抓帧分析），供监考官 scan_camera 取证
    audio: dict = {}    # 麦克风音频统计（前端 RMS 采样），供监考官 scan_audio 取证


@app.post("/api/exam/submit")
def exam_submit(body: ExamSubmitIn):
    """全卷一次 LLM 批改（逐题 0-1 分+点评）→ 逐题写答题记录并更新掌握度/Elo →
    AI防作弊监考官ReAct取证裁决 → LLM失分模式讲评 → 结构化成绩单落盘。"""
    from scholarpulse.llm import get_llm
    f = f"data/exam_{body.sid}.json"
    if not os.path.exists(f):
        raise HTTPException(400, "没有进行中的考试")
    exam = json.load(open(f, encoding="utf-8"))
    items = exam["items"]
    course = exam.get("course", "os")
    exam_mode = exam.get("mode", "fixed")
    elo_start = exam.get("elo_start", 1200)
    llm = get_llm()
    qa = [{"i": str(i), "q": it["q"], "ref": it["ref"],
           "ans": body.answers.get(str(i), "")} for i, it in enumerate(items)]
    # 作答与用时先行汇成取证日志（监考官只依赖作答+用时，不依赖评分，可与阅卷并行）
    submit_log = [{"i": i, "dur": body.timing.get(str(i), 0),
                   "ans": body.answers.get(str(i), ""), "ref": it["ref"]}
                  for i, it in enumerate(items)]
    # AI 防作弊监考官 ReAct 取证（后台线程，与阅卷并行）
    proctor_report = {"steps": [], "answer": "监考跳过", "verdict": {"risk_level": "low"}}

    def _run_proctor():
        try:
            from scholarpulse.agents.exam_proctor_agent import run_proctor
            from scholarpulse.core.curriculum import COURSES
            course_name = COURSES.get(course, course)
            cam_meta = {k: v for k, v in (body.camera or {}).items() if k != "frames"}
            cam_meta["frames_count"] = len((body.camera or {}).get("frames", []))
            aud_meta = {k: v for k, v in (body.audio or {}).items() if k != "samples_data"}
            proctor_report.update(run_proctor(body.sid, course_name, submit_log,
                                              camera=cam_meta, audio=aud_meta))
            proctor_report["camera"] = cam_meta
            proctor_report["audio"] = aud_meta
        except Exception as e:
            proctor_report.update({"answer": f"监考官失联：{e}",
                                   "verdict": {"risk_level": "unknown"}})

    import threading
    pt = threading.Thread(target=_run_proctor, daemon=True)
    pt.start()
    grades = {}
    for attempt in range(2):   # 阅卷失败自动重试一次（LLM API 抖动兜底）
        try:
            g = llm.chat_json(
                '你是阅卷老师。逐题比对作答与参考要点给分（0 到 1，要点覆盖≥60% 给 1；未作答给 0），'
                '并给一句话点评。只输出 JSON：{"grades":[{"i":"0","score":1.0,"brief":"≤15字"}]}',
                json.dumps(qa, ensure_ascii=False), temperature=0.2, max_tokens=800)
            grades = {str(x.get("i")): x for x in g.get("grades", []) if isinstance(x, dict)}
            if grades:
                break
        except Exception:
            if attempt == 0:
                import time as _t
                _t.sleep(1.5)
    mastery = storage.load_mastery(body.sid)
    elo_d = storage.load_quiz_elo(body.sid)
    tier_base = {"foundation": 1000, "consolidation": 1400, "extension": 1800}
    results, recs = [], []
    for i, it in enumerate(items):
        gr = grades.get(str(i), {})
        score = float(gr.get("score", 0)) if gr else 0.0
        correct = score >= 0.6
        ans = body.answers.get(str(i), "")
        dur = body.timing.get(str(i), 0)
        if it["cid"] in mastery:
            M.apply_answer(mastery[it["cid"]], correct)
        base = tier_base.get(it["tier"], 1200)
        expect = 1.0 / (1.0 + 10 ** ((base - elo_d["elo"]) / 400.0))
        elo_d["elo"] = max(800, elo_d["elo"] + round(32 * ((1.0 if correct else 0.0) - expect)))
        results.append({"q": it["q"], "ans": ans, "ref": it["ref"], "score": score,
                        "correct": correct, "brief": gr.get("brief", "未作答" if not ans else ""),
                        "tier": it["tier"], "cid": it["cid"]})
        recs.append(AnswerRecord(qid=it["qid"], cid=it["cid"], correct=correct,
                                 ts=storage.now_iso(), question=it["q"], reference=it["ref"]))
    elo_d["n"] += len(items)
    storage.save_mastery(body.sid, mastery)
    storage.save_quiz_elo(body.sid, elo_d)
    storage.append_answers(body.sid, recs)
    total = round(sum(r["score"] for r in results), 1)
    n_correct = sum(1 for r in results if r["correct"])
    wrongs = [r for r in results if not r["correct"]]

    # 等待并行监考官取证完成（阅卷期间已在后台运行）
    pt.join(timeout=150)
    if not isinstance(proctor_report.get("verdict"), dict):
        proctor_report["verdict"] = {"risk_level": "unknown"}

    # LLM 失分模式讲评
    review = "全卷通过，可挑战更高难度的迁移题。"
    if wrongs:
        try:
            review = llm.chat(
                "你是讲评老师。根据错题总结失分模式（≤60字，指出共同根因与下一步行动）：",
                "\n".join(f"错题:{w['q'][:40]} 参考:{w['ref'][:40]}" for w in wrongs),
                temperature=0.4, max_tokens=160)
        except Exception:
            review = f"错了 {len(wrongs)} 题，建议逐题对照参考答案复盘。"

    # 章节得分分布
    chap_scores: dict[str, list] = {}
    cmap = {c.cid: c for c in storage.load_concepts()}
    for r in results:
        ch = cmap.get(r["cid"], type("X", (), {"chapter": "未分类"})()).chapter if r["cid"] in cmap else "未分类"
        chap_scores.setdefault(ch, []).append(r["score"])
    chapter_dist = [{"chapter": ch, "avg": round(sum(v) / len(v), 2), "n": len(v)}
                    for ch, v in sorted(chap_scores.items())]

    os.remove(f)
    report = {"ts": storage.now_iso(), "course": course, "mode": exam_mode,
              "score": total, "n": len(items), "n_correct": n_correct,
              "elo_start": elo_start, "elo_end": elo_d["elo"],
              "elo_delta": elo_d["elo"] - elo_start,
              "chapter_dist": chapter_dist}
    hist_f = f"data/examhist_{body.sid}.json"
    hist = json.load(open(hist_f, encoding="utf-8")) if os.path.exists(hist_f) else []
    hist.append(report)
    json.dump(hist, open(hist_f, "w", encoding="utf-8"), ensure_ascii=False)
    return {"score": total, "n": len(items), "n_correct": n_correct,
            "results": results, "review": review,
            "elo_start": elo_start, "elo_end": elo_d["elo"],
            "elo_delta": elo_d["elo"] - elo_start,
            "chapter_dist": chapter_dist,
            "proctor": proctor_report,
            "history": hist[-8:]}


@app.get("/api/exam/result")
def exam_result(sid: str):
    """结构化成绩单：最近一次考试的各章得分/能力值变化/薄弱点LLM讲评/监考报告。"""
    hist_f = f"data/examhist_{sid}.json"
    if not os.path.exists(hist_f):
        raise HTTPException(404, "暂无考试历史")
    hist = json.load(open(hist_f, encoding="utf-8"))
    if not hist:
        raise HTTPException(404, "暂无考试历史")
    latest = hist[-1]
    # 监考报告
    proctor_f = f"data/proctor_{sid}.json"
    proctor = json.load(open(proctor_f, encoding="utf-8")) if os.path.exists(proctor_f) else None
    # 薄弱点：得分最低的章节
    weak = sorted(latest.get("chapter_dist", []),
                  key=lambda x: x["avg"])[:3]
    return {"report": latest, "weak_chapters": weak,
            "proctor_verdict": proctor.get("verdict") if proctor else None,
            "history": hist[-8:]}


class PatrolIn(BaseModel):
    sid: str
    course: str = "os"
    frame_log: list[dict] = []   # 当前及近期帧的视觉特征 [{t,bright,motion,absent,motion_high,leave}]
    events: list[dict] = []      # 近期巡逻事件 [{t,source,type,desc}]


@app.post("/api/exam/patrol")
def exam_patrol(body: PatrolIn):
    """考中实时 AI 巡逻机器人：前端检测到异常帧时上报当前帧+近期事件，
    后端用 LangChain 轻量 ReAct 快速研判是否需语音警告考生，返回精准警告话术。
    这是『和现实一样的实时监考』核心——异常发生时 LLM 实时介入分析，而非只靠前端规则。"""
    from scholarpulse.agents.exam_proctor_agent import run_patrol
    # 考试进行中校验（防止非考试期间滥用）
    ef = f"data/exam_{body.sid}.json"
    if not os.path.exists(ef):
        return {"warn": False, "text": "", "level": "normal",
                "patrol_verdict": "no_exam"}
    return run_patrol(body.sid, body.course, body.frame_log, body.events)


@app.get("/api/student/radar")
def student_radar(sid: str):
    """学生画像雷达：五维真数据（掌握度/活跃度/稳定性/进步性/广度）+ 近 14 天答题分布
    + LLM 一句话画像评语（失败降级为统计描述）。"""
    from scholarpulse.llm import get_llm
    concepts = storage.load_concepts()
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    avg = M.avg_mastery(mastery)
    n = len(answers)
    best = cur = 0
    for a in answers:
        cur = cur + 1 if a.correct else 0
        best = max(best, cur)
    half = max(len(answers) // 2, 1)

    def acc(xs):
        return sum(1 for a in xs if a.correct) / len(xs) if xs else None

    fa, ba = acc(answers[:half]), acc(answers[half:])
    prog = 0.5 + (((ba - fa) * 1.5) if fa is not None and ba is not None else 0)
    breadth = sum(1 for c in concepts if c.cid in mastery and mastery[c.cid].p >= 0.5) / max(len(concepts), 1)
    daily: dict[str, int] = {}
    for a in answers:
        d = str(a.ts)[:10]
        daily[d] = daily.get(d, 0) + 1
    dims = {"掌握度": round(avg, 2), "活跃度": round(min(n / 30, 1.0), 2),
            "稳定性": round(min(best / 10, 1.0), 2), "进步性": round(max(0, min(1, prog)), 2),
            "广度": round(breadth, 2)}
    llm = get_llm()
    try:
        comment = llm.chat(
            "你是学情分析师，用一句话（≤50字）给这个学生画像，点评最突出的长项或短板：",
            f"五维:{dims} 连对纪录:{best} 总答题:{n}", temperature=0.5, max_tokens=120)
    except Exception:
        comment = f"掌握度 {avg:.0%}，累计答题 {n} 道，最长连对 {best}。"
    return {"dims": dims, "comment": comment, "best_streak": best, "n": n,
            "daily": sorted(daily.items())[-14:]}


@app.get("/api/student/classrank")
def student_classrank(sid: str):
    """班级对比：同班（合成人口落盘数据）真实掌握度分布 → 我的排名/百分位/班均/分布直方图。"""
    import random as _rd
    students = storage.load_students()
    me = students.get(sid)
    if not me:
        raise HTTPException(400, "学生不存在")
    same = [s.sid for s in students.values()
            if getattr(s, "teacher_id", None) == getattr(me, "teacher_id", None) and s.sid != sid]
    if not same:
        same = _rd.sample([s.sid for s in students.values() if s.sid != sid], min(40, len(students) - 1))
    my_m = M.avg_mastery(storage.load_mastery(sid))
    vals = []
    for p in same[:40]:
        m = storage.load_mastery(p)
        if m:
            vals.append(M.avg_mastery(m))
    vals.append(my_m)
    rank = sum(1 for v in vals if v > my_m) + 1
    return {"my": round(my_m, 3), "class_avg": round(sum(vals) / len(vals), 3),
            "rank": rank, "n": len(vals),
            "pct": round((1 - (rank - 1) / len(vals)) * 100),
            "hist": [round(v, 2) for v in sorted(vals)]}


@app.get("/api/student/forgetting")
def student_forgetting(sid: str):
    """遗忘曲线：个人实测保持点（各考点当前掌握度 vs 距上次作答天数），
    与艾宾浩斯理论保持率曲线对照。"""
    import datetime as _dt
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    mastery = storage.load_mastery(sid)
    last_by_cid: dict[str, str] = {}
    for a in storage.load_answers(sid):
        last_by_cid[a.cid] = str(a.ts)[:10]
    now = _dt.date.today()
    pts = []
    for cid, d in last_by_cid.items():
        try:
            days = (now - _dt.date.fromisoformat(d)).days
        except ValueError:
            continue
        if cid in mastery:
            pts.append({"days": days, "ret": round(mastery[cid].p, 2), "name": name_of.get(cid, cid)})
    return {"points": pts}


# ---------- 记忆保卫战：阵地战式间隔复习（含 LangChain ReAct 防御指挥官） ----------
@app.get("/api/review/battle")
def review_battle(sid: str):
    """阵地态势：所有到期考点=失守阵地（超期天数=紧急度），并为每个阵地配一道待答题。"""
    from scholarpulse.agents.review_defender import scan_front, pick_target
    data = scan_front.invoke({"sid": sid})
    fronts = []
    for f in data["fronts"]:
        q = pick_target.invoke({"cid": f["cid"]})
        if q.get("question"):
            f["qid"] = q["qid"]
            f["question"] = q["question"]
            f["tier"] = q.get("tier", "foundation")
        fronts.append(f)
    return {"fronts": fronts, "n_lost": data["n_lost"]}


class BattleAnswerIn(BaseModel):
    sid: str
    qid: str
    answer: str


@app.post("/api/review/answer")
def review_answer(body: BattleAnswerIn):
    """收复战判定：走防御指挥官的 judge_answer 工具（LLM 判卷→掌握度/Elo/答题记录）。"""
    from scholarpulse.agents.review_defender import judge_answer
    r = judge_answer.invoke({"sid": body.sid, "qid": body.qid, "answer": body.answer})
    if "error" in r:
        raise HTTPException(400, r["error"])
    return r


@app.get("/api/review/brief")
def review_brief(sid: str):
    """LangChain ReAct 防御指挥官：自主扫描阵地→生成战术简报（返回工具轨迹）。"""
    from scholarpulse.agents.review_defender import run_defender
    try:
        return run_defender(sid)
    except Exception as e:
        raise HTTPException(500, f"指挥官失联：{e}")


@app.get("/api/exam/practice")
def exam_practice(sid: str):
    """考试闭环：从最近答错的考点（考试/练习均算）抽 3 道未做过的变式题再练。"""
    done: set[str] = set()
    wrong_cids: list[str] = []
    for a in storage.load_answers(sid)[-20:]:
        done.add(a.qid)
        if not a.correct and a.cid not in wrong_cids:
            wrong_cids.append(a.cid)
    items = []
    for q in storage.load_quiz_bank():
        if q.cid in wrong_cids and q.qid not in done:
            items.append({"qid": q.qid, "cid": q.cid, "q": q.question, "ref": q.reference_answer})
        if len(items) >= 3:
            break
    return {"items": items, "wrong_cids": wrong_cids}


# ---------- 错题检察院：ReAct 检察官对顽固错题起诉-取证-判决-提审 ----------
class RetrialIn(BaseModel):
    sid: str
    cid: str


class RetrialSubmitIn(BaseModel):
    sid: str
    qid: str
    cid: str
    answer: str


@app.get("/api/wrong/court")
def wrong_court(sid: str, fresh: int = 0):
    """检察官开庭：错题簇=未决案件。fresh=1 重新侦查（ReAct），否则读判决档案缓存。
    每案返回簇统计 + 判决（罪名/判决/处方）+ 检察官工具轨迹与结案陈词。"""
    from scholarpulse.agents.mistake_prosecutor import run_prosecutor
    clusters = student_mistakes(sid)["clusters"][:4]
    book = {v["cid"]: v for v in storage.load_prosecution(sid)}
    if fresh or any(c["cid"] not in book for c in clusters):
        try:
            briefs = run_prosecutor(sid, clusters)
        except Exception as e:
            raise HTTPException(500, f"检察官失联：{e}")
        book = {v["cid"]: v for v in storage.load_prosecution(sid)}
        for b in briefs:  # 侦查轨迹并入判决档案，缓存后仍可回看
            book.setdefault(b["cid"], {"cid": b["cid"], "concept": b["concept"]})
            book[b["cid"]].update({"steps": b["steps"], "answer": b["answer"]})
        storage.save_prosecution(sid, list(book.values()))
    cases = [{**c, **{k: v for k, v in book.get(c["cid"], {}).items()
                      if k in ("charge", "verdict", "prescription", "closed", "ts", "steps", "answer")}}
             for c in clusters]
    return {"cases": cases, "n_records": student_mistakes(sid)["n_records"]}


@app.post("/api/wrong/retrial")
def wrong_retrial(body: RetrialIn):
    """开庭提审：从题库抽该考点学生没做过的题当庭重审（全做过则重做一道）。"""
    done = {a.qid for a in storage.load_answers(body.sid)}
    pool = [q for q in storage.load_quiz_bank() if q.cid == body.cid]
    if not pool:
        raise HTTPException(400, "题库中无该考点题目")
    q = (next((x for x in pool if x.qid not in done), pool[0]))
    return {"qid": q.qid, "cid": q.cid, "question": q.question, "tier": q.tier}


@app.post("/api/wrong/retrial/submit")
def wrong_retrial_submit(body: RetrialSubmitIn):
    """提审判卷：复用防御指挥官的 judge_answer（LLM 判卷→掌握度/Elo/答题记录）；
    答对则该案标记康复结案。"""
    from scholarpulse.agents.review_defender import judge_answer
    r = judge_answer.invoke({"sid": body.sid, "qid": body.qid, "answer": body.answer})
    if "error" in r:
        raise HTTPException(400, r["error"])
    closed = False
    if r["correct"]:
        book = storage.load_prosecution(body.sid)
        for v in book:
            if v["cid"] == body.cid and not v.get("closed"):
                v["closed"] = True
                v["prescription"] = (v.get("prescription", "") + " ｜ 提审通过，当庭结案").strip("｜ ")
                closed = True
        storage.save_prosecution(body.sid, book)
    return {**r, "closed": closed}


@app.get("/api/report")
def report(sid: str):
    concepts = storage.load_concepts()
    student = storage.load_students()[sid]
    mastery = storage.load_mastery(sid)
    snaps = storage.load_snapshots(sid)
    prev = snaps[-2] if len(snaps) >= 2 else None
    cur = {c.cid: c.p for c in mastery.values()}
    rep = report_agent.student_report(concepts, prev, cur, student.name)
    storage.append_snapshot(sid, ReportSnapshot(ts=storage.now_iso(), mastery=cur))
    return {"report": rep, "exam_oracle": meta_agent.exam_oracle(concepts),
            "wrong_book": quiz_bank.wrong_book(storage.load_answers(sid))}


@app.get("/api/advanced")
def advanced(sid: str):
    from scholarpulse.core import advanced as adv
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    records = storage.load_answers(sid)
    cm = adv.confusion_matrix(records, concepts)
    dl = next((c.cid for c in concepts if "死锁" in c.name), concepts[0].cid if concepts else "")
    chain = adv.root_cause(dl, mastery, concepts) if dl else []
    ci = {cid: list(adv.confidence_interval(m.p, m.n_correct + m.n_wrong))
          for cid, m in list(mastery.items())[:6]}
    return {"confusion": cm, "root_cause_chain": chain, "confidence_interval": ci}


class FeynmanIn(BaseModel):
    sid: str
    concept: str
    explain: str


@app.post("/api/feynman")
def feynman(body: FeynmanIn):
    from scholarpulse.core import advanced as adv
    return adv.feynman_check(body.concept, body.explain)


class ChatIn(BaseModel):
    sid: str
    message: str


class DebateIn(BaseModel):
    sid: str
    question: str
    answer: str


@app.post("/api/debate-grade")
def debate_grade(req: DebateIn):
    from scholarpulse.agents import debate_grading_agent
    return debate_grading_agent.debate_grade(req.question, req.answer)


# ---------- 数字孪生对抗赛 ----------
def _twin_score(battles: list[TwinBattle]) -> dict:
    return {"student": sum(1 for b in battles if b.winner == "student"),
            "twin": sum(1 for b in battles if b.winner == "twin"),
            "tie": sum(1 for b in battles if b.winner == "tie"),
            "total": len(battles)}


@app.get("/api/twin/profile")
def twin_profile(sid: str):
    """孪生画像：不存在或新增答题记录≥3条则进化重建。"""
    _ensure_student(sid)
    twin = storage.load_twin(sid)
    prof, battles = twin["profile"], twin["battles"]
    n_ans = len(storage.load_answers(sid))
    if prof is None or (n_ans - prof.base_n) >= 3:
        prof = twin_agent.build_profile(sid)
        prof.base_n = n_ans
        prof.similarity = twin_agent.calc_similarity(battles)["sim"]
        storage.save_twin(sid, prof, battles)
    return {"profile": prof.model_dump(), "cold": n_ans == 0, "score": _twin_score(battles)}


class TwinStartIn(BaseModel):
    sid: str


@app.post("/api/twin/battle/start")
def twin_battle_start(body: TwinStartIn):
    """红方针对孪生盲区出诱饵题（过反幻觉校验），参考答案只存服务端 pending。"""
    _ensure_student(body.sid)
    twin = storage.load_twin(body.sid)
    prof = twin["profile"]
    if prof is None:
        prof = twin_agent.build_profile(body.sid)
        prof.base_n = len(storage.load_answers(body.sid))
    concepts = storage.load_concepts()
    if not concepts:
        return {"done": True, "msg": "请先上传课件构建知识图谱"}
    mastery = storage.load_mastery(body.sid)
    # 目标概念：优先孪生误区所在考点，否则掌握度最低者
    target = next((next(c for c in concepts if c.cid == m.cid) for m in prof.misconceptions
                   if m.cid in {c.cid for c in concepts}), None)
    if target is None:
        weak = M.weak_concepts(mastery, concepts)
        target = weak[0] if weak else concepts[0]
    bait = twin_agent.make_bait(prof, target, _material_of(body.sid))
    if not bait:
        return {"done": True, "msg": "红方出题暂不可用（LLM不可用），稍后再试。"}
    item = bait["item"]
    storage.save_twin_pending(body.sid, {"qid": item.qid, "cid": item.cid,
                                         "question": item.question,
                                         "reference": item.reference_answer,
                                         "trick": bait.get("trick", ""),
                                         "profile": prof.model_dump()})
    return {"done": False, "qid": item.qid, "cid": item.cid, "question": item.question}


class TwinSubmitIn(BaseModel):
    sid: str
    qid: str
    student_answer: str
    confidence: int = 3


@app.post("/api/twin/battle/submit")
def twin_battle_submit(body: TwinSubmitIn):
    """学生先交卷，孪生再盲答（签名不含学生答案，双盲），双批改后揭晓。"""
    pend = storage.load_twin_pending(body.sid)
    if not pend:
        raise HTTPException(400, "没有待作答的孪生对战题")
    if pend["qid"] != body.qid:
        raise HTTPException(400, "题目编号不匹配，请刷新页面")
    prof = TwinProfile(**pend["profile"])
    # 双盲：学生答案已入局部变量，孪生作答接口物理上拿不到它
    ta = twin_agent.twin_answer(prof, pend["question"])
    sg = meta_agent.grade_answer(pend["question"], pend["reference"], body.student_answer)
    tg = meta_agent.grade_answer(pend["question"], pend["reference"], ta.get("answer", ""))
    s_ok = float(sg.get("score", 0)) >= 0.6
    t_ok = float(tg.get("score", 0)) >= 0.6
    winner = twin_agent.judge_winner(s_ok, t_ok)
    rev = twin_agent.reveal_analysis(pend["question"], pend["reference"],
                                     body.student_answer, ta.get("answer", ""), prof)
    twin = storage.load_twin(body.sid)
    battle = TwinBattle(bid=f"tb_{len(twin['battles']) + 1}_{int(time.time())}",
                        qid=pend["qid"], cid=pend["cid"], question=pend["question"],
                        reference=pend["reference"], student_answer=body.student_answer,
                        student_conf=body.confidence, twin_answer=ta.get("answer", ""),
                        twin_conf=int(ta.get("confidence", 3)), s_correct=s_ok,
                        t_correct=t_ok, winner=winner, trick=pend.get("trick", ""),
                        analysis=rev.get("analysis", ""), ts=storage.now_iso())
    battles = twin["battles"] + [battle]
    sim = twin_agent.calc_similarity(battles)
    prof.similarity = sim["sim"]
    prof.updated_at = storage.now_iso()
    storage.save_twin(body.sid, prof, battles)
    storage.clear_twin_pending(body.sid)
    return {"battle": battle.model_dump(), "sim": sim, "score": _twin_score(battles),
            "comment_s": sg.get("comment", ""), "comment_t": tg.get("comment", ""),
            "insight": rev.get("insight", "")}


@app.get("/api/twin/state")
def twin_state(sid: str):
    """恢复中断的对战（防刷新丢题）+ 比分板与相似度趋势。"""
    _ensure_student(sid)
    pend = storage.load_twin_pending(sid)
    twin = storage.load_twin(sid)
    battles = twin["battles"]
    return {"pending": ({"qid": pend["qid"], "cid": pend["cid"],
                         "question": pend["question"]} if pend else None),
            "score": _twin_score(battles),
            "sim": twin_agent.calc_similarity(battles),
            "has_profile": twin["profile"] is not None}


@app.get("/api/diagnose")
def diagnose(sid: str):
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    weak = M.weak_concepts(mastery, concepts)
    if not weak:
        return {"done": True, "msg": "无薄弱点"}
    target = weak[0]
    cmap = {c.cid: c for c in concepts}
    # 沿前置链回溯，形成诊断路径
    chain, seen = [], set()
    cur = target.cid
    while cur and cur not in seen and cur in cmap:
        seen.add(cur)
        c = cmap[cur]
        chain.append({"cid": c.cid, "name": c.name, "mastery": round(mastery[c.cid].p, 2) if c.cid in mastery else None})
        cur = c.prerequisites[0] if c.prerequisites else None
    # 找掌握度最低的环节 = 断层根
    broken = min(chain, key=lambda x: x["mastery"] if x["mastery"] is not None else 1)
    return {"target": target.name, "chain": chain, "root_break": broken}


@app.get("/api/reflect-quiz")
def reflect_quiz(sid: str):
    from scholarpulse.agents import reflect_loop_agent
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    weak = M.weak_concepts(mastery, concepts)
    if not weak:
        return {"done": True}
    target = weak[0]
    material = _material_of(sid)
    return reflect_loop_agent.reflective_quiz(f"{target.name}:{target.summary}", material, max_rounds=3)


@app.post("/api/orchestrate")
def orchestrate(req: ChatIn):
    """统一总控智能体入口（多端异构）：真正的 LangChain 工具调用 agent。"""
    from scholarpulse.agents import tool_calling_agent
    return tool_calling_agent.run(req.message, req.sid)


@app.post("/api/chat")
def chat(body: ChatIn):
    """AI 助教：LangChain ReAct + 项目知识库/课件 RAG，返回答案、工具轨迹与引用。"""
    from scholarpulse.agents.companion_agent import run_companion
    r = run_companion(body.sid, body.message)
    return {"answer": r["answer"], "tool": " → ".join(s["tool"] for s in r["steps"]) or "chat",
            "steps": r["steps"], "citations": r.get("citations", []), "web": r.get("web", [])}


# ---------- 学生端诊断智能体矩阵（全部 create_react_agent） ----------
@app.get("/api/root/trace")
def root_trace(sid: str):
    """🌳 根因侦探（ReAct）：对最薄弱考点立案，查先修链+错误时序，宣布真凶并归档。"""
    from scholarpulse.agents.root_detective import run_detective
    r = run_detective(sid)
    if not r["steps"]:
        raise HTTPException(500, r["answer"])
    return r


@app.get("/api/oracle/predict")
def oracle_predict(sid: str):
    """🔮 押题先知（ReAct）：三路证据交叉研判（班级共性×个人错题×考试权重），落盘押题清单。"""
    from scholarpulse.agents.exam_oracle_agent import run_oracle
    r = run_oracle(sid)
    if not r["steps"]:
        raise HTTPException(500, r["answer"])
    return r


@app.get("/api/diagnose/ct")
def diagnose_ct(sid: str, target: str = ""):
    """🩺 断层 CT 室（ReAct）：沿先修链扫描真断点，开处方落盘。"""
    from scholarpulse.agents.diagnosis_agents import run_ct
    r = run_ct(sid, target)
    if not r["steps"]:
        raise HTTPException(500, r["answer"])
    return r


@app.get("/api/advanced/forensic")
def advanced_forensic(sid: str):
    """🧠 认知法医（ReAct）：全景尸检，出具法医鉴定书落盘。"""
    from scholarpulse.agents.diagnosis_agents import run_forensic
    r = run_forensic(sid)
    if not r["steps"]:
        raise HTTPException(500, r["answer"])
    return r


# ---------- 博士级创新：四大学术前沿 ----------

class DefenseIn(BaseModel):
    sid: str
    course: str = "os"
    qa_set: list[dict]   # [{"q":题干,"ans":学生答,"ref":参考答案}]


@app.post("/api/exam/defense")
def exam_defense(body: DefenseIn):
    """🎓 博士答辩委员会（Multi-Agent Debate）：3派考官独立评分→2轮交叉辩论→主持人裁决。"""
    from scholarpulse.agents.thesis_defense_agent import run_thesis_defense
    r = run_thesis_defense(body.sid, body.course, body.qa_set)
    return r


@app.get("/api/exam/generative")
def exam_generative(sid: str, course: str = "os"):
    """🧬 生成式CAT（Generative Assessment）：LLM实时生成新题+反幻觉grounding校准+IRT难度校准。"""
    from scholarpulse.agents.generative_cat import run_generative_cat
    r = run_generative_cat(sid, course)
    if not r["items"]:
        raise HTTPException(500, r.get("answer", "生成失败"))
    return r


@app.get("/api/diagnosis/causal")
def diagnosis_causal(sid: str, cid: str = ""):
    """🔮 因果推断（do-calculus）：沿因果图溯源真根因，区分因果与伪相关，出诊断书。"""
    from scholarpulse.agents.causal_inference_agent import run_causal_diagnosis
    r = run_causal_diagnosis(sid, cid)
    if not r["steps"]:
        raise HTTPException(500, r["answer"])
    return r


@app.get("/api/diagnosis/reflexion")
def diagnosis_reflexion(sid: str, target: str = ""):
    """🔬 Reflexion科学方法循环（CT）：假设→插桩→复现→分析→修正→验证，带反思记忆多轮进化。"""
    from scholarpulse.agents.reflexion_agent import run_ct_reflexion
    r = run_ct_reflexion(sid, target)
    if not r["steps"]:
        raise HTTPException(500, r["answer"])
    return r


@app.get("/api/diagnosis/forensic/reflexion")
def diagnosis_forensic_reflexion(sid: str):
    """🔬 Reflexion科学方法循环（法医）：多轮假设-取证-反思-修正的自我进化尸检。"""
    from scholarpulse.agents.reflexion_agent import run_forensic_reflexion
    r = run_forensic_reflexion(sid)
    if not r["steps"]:
        raise HTTPException(500, r["answer"])
    return r


@app.get("/api/quiz/brief")
def quiz_brief(sid: str):
    """✍️ 考官开局（ReAct）：查 Elo/薄弱/错题本后下发今日训练策略。"""
    from scholarpulse.agents.companion_agent import run_examiner
    r = run_examiner(sid)
    return {"answer": r["answer"], "steps": r["steps"]}


@app.get("/api/teacher/class")
def teacher_class(tid: str):
    """教师班级看板（ReAct 智能体版）：
    🏫 班级教研官自主 scan_class_distribution → scan_concept_failure_clusters →
    diagnose_at_risk_students → 推理班级画像+分层干预处方。
    劝学信由教研官基于完整证据生成，不再单独调 meta_agent。"""
    from scholarpulse.agents.teacher_intelligence_agent import run_class_teacher
    from scholarpulse import storage, mastery as M
    from scholarpulse.llm import get_llm
    # ReAct 智能体主输出
    react = run_class_teacher(tid)
    # 保留名册供教研会诊下拉（这部分是展示数据，不需要 LLM）
    t = storage.load_teachers().get(tid)
    students = storage.load_students()
    pool = {sid: s for sid, s in students.items() if s.teacher_id == tid} or students
    roster = sorted(({"sid": sid, "name": s.name,
                      "avg": round(M.avg_mastery(storage.load_mastery(sid)), 2)}
                     for sid, s in pool.items()), key=lambda r: r["avg"])
    # 劝学信：直接调 diagnose_at_risk_students 拿完整证据（不走 ReAct 截断的 steps）
    letters = []
    from scholarpulse.agents.teacher_intelligence_agent import diagnose_at_risk_students
    try:
        risk_json = diagnose_at_risk_students.invoke({"tid": tid})
    except Exception:
        risk_json = None
    if risk_json:
        llm = get_llm()
        for stu in risk_json.get("students", [])[:3]:
            name = stu.get("name", "")
            avg = stu.get("avg_mastery", 0)
            root = stu.get("root_cause", "")
            stuck = stu.get("stuck_concept") or {}
            stuck_name = stuck.get("name", "")
            try:
                letter_text = llm.chat_json(
                    "你是一位关心学生的大学班主任，写一封30-50字的个性化劝学信。"
                    "称呼同学名，点出具体根因（如卡壳的考点），给出一个具体行动。不要说教，要真诚。",
                    f"学生：{name}；平均掌握度：{avg}；根因：{root}；卡壳考点：{stuck_name}"
                ).get("letter", "")
            except Exception:
                letter_text = f"同学你好，根据本班学情画像，你的主要问题是{root}，建议今晚花30分钟重点攻克{stuck_name or '核心概念'}。"
            letters.append({"to": name, "letter": letter_text, "avg": avg})
    # 兼容旧接口字段（board 用于年级总览等硬编码字段，已替换为 react.answer）
    return {"react": react, "intervention_letters": letters,
            "student_count": len(pool),
            "teacher": {"name": t.name, "subject": t.subject} if t else None,
            "roster": roster,
            # 保留 board 字段兼容前端旧渲染（前端已改渲染 react）
            "board": {"class_weak_points": [], "at_risk_students": [],
                      "teaching_advice": react.get("answer", "")[:200]}}


@app.get("/api/teacher/grade-insight")
def teacher_grade_insight(tid: str):
    """年级督导员（ReAct 智能体）：🏛️ scan_grade_dispersion → scan_bottleneck_concepts →
    prescribe_grade_intervention → 推理本班在年级的定位 + 干预处方。"""
    from scholarpulse.agents.teacher_intelligence_agent import run_grade_supervisor
    return run_grade_supervisor(tid)


class ConsultIn(BaseModel):
    sid: str


@app.get("/api/llm/status")
def llm_status():
    """多供应商 LLM 状态：DeepSeek→千问→豆包故障转移链 + 各家调用统计（前端徽章）。"""
    from scholarpulse.llm import get_llm
    return get_llm().status()


@app.get("/api/student/mistakes")
def student_mistakes(sid: str):
    """错题聚类：按考点聚合真实答题记录中的错题（次数/正确率），错题本与聚类的数据源。"""
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    acc: dict[str, dict] = {}
    for a in storage.load_answers(sid):
        g = acc.setdefault(a.cid, {"cid": a.cid, "wrong": 0, "total": 0,
                                   "last_q": "", "last_ref": ""})
        g["total"] += 1
        if not a.correct:
            g["wrong"] += 1
            if a.question:
                g["last_q"], g["last_ref"] = a.question, a.reference
    rows = sorted(({"concept": name_of.get(k, k), **v,
                    "wrong_rate": round(v["wrong"] / v["total"], 2)}
                   for k, v in acc.items() if v["wrong"] > 0),
                  key=lambda r: -r["wrong"])
    return {"clusters": rows, "n_records": sum(g["total"] for g in acc.values())}


@app.get("/api/student/mistakes/clinic")
def mistake_clinic(sid: str):
    """错题病理科：把错题簇升级为「病灶」——
    ①感染链：某考点的前置概念也在错题列表 → 标记根因在上游（先治上游，下游或自愈）；
    ②复发风险指数 0-99：错误率×未掌握度×错误频次加权合成；
    ③LLM 诊断病灶类型（概念混淆/技能缺失/审题习惯/记忆模糊）并开 3 天治疗处方
      （只对高风险或根因簇调用 LLM，失败降级为规则模板）。"""
    from scholarpulse.llm import get_llm
    concepts = storage.load_concepts()
    cid_of = {c.name: c.cid for c in concepts}
    name_of = {c.cid: c.name for c in concepts}
    # prerequisites 存的是 cid，先翻译成考点名再与错题簇（按名字）比对
    prereq_of = {c.name: [name_of.get(p, p) for p in c.prerequisites] for c in concepts}
    base = student_mistakes(sid)
    mastery = storage.load_mastery(sid)
    llm = get_llm()
    rows = base["clusters"]
    names = {r["concept"] for r in rows}
    for r in rows:
        r["upstream"] = [p for p in prereq_of.get(r["concept"], []) if p in names]
        r["is_root"] = not r["upstream"]
        cid = cid_of.get(r["concept"])
        mp = mastery[cid].p if cid in mastery else 0.5
        r["risk"] = int(min(99, r["wrong_rate"] * 60 + (1 - mp) * 25 + min(r["wrong"], 5) * 3))
    targets = [r for r in rows if r["is_root"] or r["risk"] >= 40][:4]
    for r in targets:
        try:
            d = llm.chat_json(
                '你是错题病理科医生。诊断该病灶类型（四选一：概念混淆/技能缺失/审题习惯/记忆模糊），'
                '并开 3 天治疗处方（每天一句话：复习什么/做什么/自测标准）。只输出 JSON：'
                '{"type":"四选一","prescription":["第1天…","第2天…","第3天…"]}',
                f"考点:{r['concept']} 错{r['wrong']}/{r['total']}次 错误率{r['wrong_rate']:.0%} "
                f"最近错题:{r['last_q'][:60]}", temperature=0.4, max_tokens=300)
            r["pathology"] = d.get("type", "概念混淆")
            r["prescription"] = [str(x) for x in d.get("prescription", [])][:3]
        except Exception:
            r["pathology"], r["prescription"] = "概念混淆", [f"重读{r['concept']}讲义并复述定义",
                                                            f"重做错题：{r['last_q'][:30]}", "自测一道变式题"]
    for r in rows:
        if "pathology" not in r:
            r["pathology"], r["prescription"] = "", []
    return {"clusters": rows, "n_records": base["n_records"]}


@app.get("/api/student/badges")
def student_badges(sid: str):
    """成就徽章：基于真实数据的规则引擎（答题量/连胜/掌握里程碑/孪生战绩/错题攻克）。"""
    from scholarpulse.mastery import avg_mastery
    answers = storage.load_answers(sid)
    mastery = storage.load_mastery(sid)
    students = storage.load_students()
    s = students.get(sid)
    n = len(answers)
    streak_best, cur = 0, 0
    for a in answers:
        cur = cur + 1 if a.correct else 0
        streak_best = max(streak_best, cur)
    avg = avg_mastery(mastery)
    twin = storage.load_twin(sid)
    wins = (twin.get("score") or {}).get("student", 0) if isinstance(twin, dict) else 0
    got = []
    for key, cond, label in [
        ("first", n >= 1, "初试锋芒：完成第 1 道题"),
        ("ten", n >= 10, "小有所成：完成 10 道题"),
        ("fifty", n >= 50, "题海老兵：完成 50 道题"),
        ("streak3", streak_best >= 3, "三连击：连续答对 3 题"),
        ("streak5", streak_best >= 5, "五连登科：连续答对 5 题"),
        ("half", avg >= 0.5, "过半山河：平均掌握度 ≥50%"),
        ("master", avg >= 0.8, "融会贯通：平均掌握度 ≥80%"),
        ("twin1", wins >= 1, "知己者胜：孪生对抗赢 1 局"),
        ("twin3", wins >= 3, "洞若观火：孪生对抗赢 3 局"),
    ]:
        if cond:
            got.append({"key": key, "label": label})
    return {"badges": got, "n_answers": n, "best_streak": streak_best,
            "avg": round(avg, 2), "twin_wins": wins,
            "name": s.name if s else sid}


@app.get("/api/student/cogload")
def student_cogload(sid: str):
    """认知负荷与学习情绪推断：把握度校准偏差 + 复习到期密度 + 近期正确率走势（规则合成）。"""
    from scholarpulse.mastery import avg_mastery
    answers = storage.load_answers(sid)[-10:]
    mastery = storage.load_mastery(sid)
    due = len([1 for m in mastery.values() if getattr(m, "due_days", 99) <= 0]) \
        if mastery and hasattr(next(iter(mastery.values())), "due_days") else 0
    if answers:
        acc = sum(1 for a in answers if a.correct) / len(answers)
        trend = ("上升" if acc >= 0.6 else "波动" if acc >= 0.4 else "下滑")
        load = min(1.0, (1 - acc) * 0.6 + min(len(answers), 10) / 20)
        mood = ("专注" if acc >= 0.7 else "平稳" if acc >= 0.5 else "受挫" if acc < 0.35 else "波动")
    else:
        acc, trend, load, mood = None, "暂无数据", 0.2, "平静"
    return {"recent_accuracy": round(acc, 2) if acc is not None else None,
            "trend": trend, "cognitive_load": round(load, 2), "mood": mood,
            "avg_mastery": round(avg_mastery(mastery), 2) if mastery else 1.0,
            "n_recent": len(answers)}


# ---------- 🧠 学情洞察 ReAct 智能体矩阵（8 个面板全部 LangChain ReAct） ----------
@app.get("/api/insight/star-map")
def insight_star_map(sid: str):
    """🌟 星图制图师（ReAct）：拓扑分析 + 学习路径导航。"""
    from scholarpulse.agents.study_insight_agents import run_star_cartographer
    return run_star_cartographer(sid)


@app.get("/api/insight/mood")
def insight_mood(sid: str):
    """😊 情绪观察员（ReAct）：答题时序推断情绪 + 干预建议。"""
    from scholarpulse.agents.study_insight_agents import run_mood_observer
    return run_mood_observer(sid)


@app.get("/api/insight/profile")
def insight_profile(sid: str):
    """🧭 画像鉴定师（ReAct）：五维深度画像 + 鉴定书。"""
    from scholarpulse.agents.study_insight_agents import run_profile_appraiser
    return run_profile_appraiser(sid)


@app.get("/api/insight/load")
def insight_load(sid: str):
    """🧠 负荷监测官（ReAct）：负荷校准 + 调节处方。"""
    from scholarpulse.agents.study_insight_agents import run_load_monitor
    return run_load_monitor(sid)


@app.get("/api/insight/class")
def insight_class(sid: str):
    """🏫 班级情报官（ReAct）：全班分布 + 共性弱点简报。"""
    from scholarpulse.agents.study_insight_agents import run_class_intel
    return run_class_intel(sid)


@app.get("/api/insight/forgetting")
def insight_forgetting(sid: str):
    """📉 遗忘预测师（ReAct）：衰减拟合 + 预警。"""
    from scholarpulse.agents.study_insight_agents import run_forgetting_forecaster
    return run_forgetting_forecaster(sid)


@app.get("/api/insight/courseware")
def insight_courseware(sid: str):
    """📑 课件拆解师（ReAct）：上传后智能拆解 + 学习指南。"""
    from scholarpulse.agents.study_insight_agents import run_courseware_deconstructor
    return run_courseware_deconstructor(sid)


@app.get("/api/rag/search")
def rag_search(sid: str, q: str, remote: bool = False):
    """🔍 混合检索 RAG + GraphRAG：课件原文『密集向量+CSR稀疏矩阵BM25』双路召回
    + RRF 融合 + 先修图扩展。remote=true 时叠加硅基流动语义嵌入第三通道
    （失败自动降级离线 2 通道）。"""
    from scholarpulse.core.rag_retriever import get_student_retriever, expand_with_graph
    r, meta = get_student_retriever(sid)
    if r is None:
        return meta
    if remote:
        from scholarpulse.core.siliconflow_embed import hybrid_search
        out = hybrid_search(sid, q, top_k=5, retriever=r)
        hits = out["hits"]
        meta = {**meta, "channels": out["channels"], **{k: v for k, v in out.items()
                                                        if k in ("embed_model",)}}
    else:
        hits = r.search(q, top_k=5)
    return {"source": meta.get("source"), "n_chunks": meta.get("n_chunks"),
            "channels": meta.get("channels", 2), "query": q, "hits": hits,
            "graph_context": expand_with_graph(
                hits, storage.load_concepts(), storage.load_mastery(sid))}


@app.post("/api/rag/index")
async def rag_index(sid: str, req: Request):
    """✍️ 索引回写（双向数据流·写方向）：把 LLM 生成知识（摘要/补弱建议）写入检索索引，
    下一次检索即可命中。body: {"texts": ["...", ...]}"""
    from scholarpulse.core.rag_retriever import index_llm_knowledge
    body = await req.json()
    texts = body.get("texts") or []
    if not isinstance(texts, list) or not texts:
        return {"error": "texts 需为非空字符串数组"}
    return index_llm_knowledge(sid, [str(t) for t in texts])


@app.get("/api/rag/eval")
def rag_eval():
    """📏 检索质量评测：固定评测集上量化 dense/sparse/hybrid 三路的 hit@k、MRR 与检索延迟。"""
    import time as _t
    from scholarpulse.core.rag_eval import run_eval, EVAL_CORPUS, EVAL_QUERIES
    from scholarpulse.core.rag_retriever import HybridRetriever, chunk_text
    out = run_eval()
    chunks = chunk_text(EVAL_CORPUS, size=110, overlap=15)
    ret = HybridRetriever(chunks)
    samples = []
    for _ in range(5):
        for q in EVAL_QUERIES:
            t0 = _t.perf_counter()
            ret.search(q["q"], top_k=len(chunks))
            samples.append((_t.perf_counter() - t0) * 1000)
    samples.sort()
    n = len(samples)
    out["latency"] = {"runs": n,
                      "avg_ms": round(sum(samples) / n, 2),
                      "p50_ms": round(samples[n // 2], 2),
                      "p95_ms": round(samples[min(int(n * 0.95), n - 1)], 2)}
    return out


@app.get("/gen-eval")
def gen_eval_page():
    """生成质量抽检页：引用 / 拒答 / 不编造 / 贴题 四维判定。"""
    return FileResponse(os.path.join(STATIC, "gen-eval.html"))


_GEN_CASES = [
    {"q": "遗忘曲线的复习间隔怎么安排", "expect": "answer"},
    {"q": "TCP 三次握手的过程是什么", "expect": "answer"},
    {"q": "量子隧穿效应是什么", "expect": "refuse"},
    {"q": "怎么调用 /api/magic/export 导出成绩", "expect": "no_fabricate"},
    {"q": "我们班本学期期末的平均分和及格率是多少", "expect": "refuse"},
    {"q": "ScholarPulse 的 AI 自动阅卷功能怎么用", "expect": "refuse"},
]


@app.get("/api/gen/eval")
def gen_eval_api():
    """生成质量抽检：固定 6 类用例，按引用/拒答/编造声明/贴题四维规则判定（零人工评分）。"""
    import re
    import time as _t
    from scholarpulse.core.rag_eval import EVAL_CORPUS
    from scholarpulse.core.rag_retriever import HybridRetriever, chunk_text
    from scholarpulse.core.grounding import check as ground_check
    from scholarpulse.llm import get_llm

    src = open(os.path.join(os.path.dirname(__file__), "app.py"), encoding="utf-8").read()
    routes = {r.rstrip("/") for r in
              re.findall(r'@app\.(?:get|post|put|delete)\("([^"]+)"', src)}

    chunks = chunk_text(EVAL_CORPUS, size=110, overlap=15)
    ret = HybridRetriever(chunks)
    sys_p = ("你是课程助教。只能依据给定资料回答；资料不足时必须明确说明“资料中没有相关内容”，"
             "禁止编造接口、参数或事实。引用资料时标注 [片段n]。")
    REFUSE = ("没有", "未找到", "资料中", "无法", "不足")
    STOP = {"什么", "怎么", "如何", "有哪", "哪些"}

    cases, passed = [], 0
    for c in _GEN_CASES:
        t0 = _t.perf_counter()
        hits = ret.search(c["q"], top_k=3)
        ctx = "\n".join(f"[片段{i + 1}] {h['text']}" for i, h in enumerate(hits)) or "（无）"
        try:
            ans = get_llm().chat(sys_p, f"资料：\n{ctx}\n\n问题：{c['q']}",
                                 temperature=0.2, max_tokens=300)
        except Exception as e:
            ans = f"（模型不可用：{e}）"
        ms = round((_t.perf_counter() - t0) * 1000)

        q = c["q"]
        grams = {q[i:i + 2] for i in range(len(q) - 1)} - STOP
        cited = bool(re.search(r"\[片段\d+\]", ans))
        refused = any(k in ans for k in REFUSE)
        # 代码级反编造校验：答案里的数字/接口路径必须能在检索证据中落地
        g = ground_check(ans, [{"text": ctx}], question=q, exclude=routes)
        fabricated = [x for x in g["unverified"] if x.startswith("/")]
        topic = round(sum(1 for gr in grams if gr in ans) / max(len(grams), 1), 2)

        if c["expect"] == "answer":
            ok = (not refused) and topic >= 0.3
        elif c["expect"] == "refuse":
            ok = refused
        else:
            ok = not fabricated
        ok = ok and g["ok"]
        passed += 1 if ok else 0
        cases.append({"q": q, "expect": c["expect"], "answer": ans[:400],
                      "cited": cited, "refused": refused, "topic": topic,
                      "fabricated": fabricated, "unverified": g["unverified"],
                      "claims": g["n_claims"], "ok": ok, "ms": ms})

    return {"total": len(cases), "passed": passed,
            "rate": round(passed / max(len(cases), 1), 2),
            "no_fabricate": all(not x["fabricated"] for x in cases),
            "grounded": all(not x["unverified"] for x in cases),
            "unverified_total": sum(len(x["unverified"]) for x in cases),
            "avg_ms": round(sum(x["ms"] for x in cases) / max(len(cases), 1)),
            "cases": cases}


@app.get("/models")
def models_page():
    """模型配置页：供应商切换、配置信息、余额与 token 用量。"""
    return FileResponse(os.path.join(STATIC, "models.html"))


class ModelSwitchIn(BaseModel):
    name: str


@app.get("/api/llm/config")
def llm_config():
    from scholarpulse.llm import get_llm
    return get_llm().config()


@app.post("/api/llm/switch")
def llm_switch(body: ModelSwitchIn):
    from scholarpulse.llm import get_llm
    try:
        return get_llm().set_active(body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/llm/usage")
def llm_usage():
    from scholarpulse.llm import get_llm
    return get_llm().usage()


@app.get("/api/llm/quota")
def llm_quota(name: str = ""):
    from scholarpulse.llm import get_llm
    return {"items": get_llm().quota(name or None)}


class ProviderIn(BaseModel):
    name: str
    base: str
    model: str
    key: str
    timeout_ms: int = 10000


class ModelTestIn(BaseModel):
    name: str = ""


@app.post("/api/llm/test")
def llm_test(body: ModelTestIn):
    """真实连通性测试：直接请求目标供应商，返回延迟、回复与 token。"""
    from scholarpulse.llm import get_llm
    c = get_llm()
    try:
        return c.probe(body.name or c.active)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/llm/presets")
def llm_presets():
    """预置供应商目录：填了密钥就能一键盘启用。"""
    from scholarpulse.config import PRESETS, settings
    names = {p.name for p in settings.providers}
    return {"items": [{"name": k.lower(), "base": v[0], "model": v[1],
                       "ready": k.lower() in names} for k, v in PRESETS.items()]}


@app.post("/api/llm/add")
def llm_add(body: ProviderIn):
    """新增任意 OpenAI 兼容模型：立即可用，落盘 data/llm_providers.json。"""
    from scholarpulse.config import add_runtime_provider
    from scholarpulse.llm import get_llm
    try:
        p = add_runtime_provider(body.name, body.base, body.model, body.key,
                                 body.timeout_ms)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"name": p.name, "config": get_llm().config()}


@app.delete("/api/llm/provider/{name}")
def llm_remove(name: str):
    from scholarpulse.config import remove_runtime_provider
    from scholarpulse.llm import get_llm
    try:
        remove_runtime_provider(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    c = get_llm()
    if c.active == name.lower() and c.providers:
        c.set_active(c.providers[0].name)
    return c.config()


@app.get("/api/graph/neighbors")
def graph_neighbors(sid: str, cid: str):
    """🕸 GraphRAG 邻居扩展：指定概念沿先修图的一跳邻居（上游先修/下游依赖+掌握度）。"""
    from scholarpulse.core.rag_retriever import expand_with_graph
    concepts = storage.load_concepts()
    center = next((c for c in concepts if c.cid == cid), None)
    if center is None:
        raise HTTPException(404, f"概念不存在：{cid}")
    mastery = storage.load_mastery(sid)
    neighbors = expand_with_graph([{"text": center.name, "rank": 1}],
                                  concepts, mastery, max_neighbors=8)
    return {"center": {"cid": center.cid, "name": center.name,
                       "mastery": mastery.get(cid).p if mastery.get(cid) else None},
            "neighbors": neighbors}


# ---------- 学习管理可视化：思维导图 / 饼图 / 笔记 / 多格式导出 ----------
@app.get("/api/mindmap")
def mindmap(sid: str):
    """🌳 思维导图树：章节→概念（掌握度）→补弱建议，层层推进。"""
    from scholarpulse.core.report_exporter import build_mindmap
    return build_mindmap(storage.load_concepts(), storage.load_mastery(sid), sid)


@app.get("/api/pie")
def pie(sid: str):
    """🥧 掌握度三分层饼图数据（基础/巩固/拓展，比例归一）。"""
    from scholarpulse.core.report_exporter import build_pie
    return build_pie(storage.load_mastery(sid))


@app.get("/api/notes")
def notes_get(sid: str):
    """📝 学生笔记列表。"""
    return storage.load_notes(sid)


class NoteIn(BaseModel):
    sid: str
    text: str


@app.post("/api/notes")
def notes_post(body: NoteIn):
    """📝 新增学生笔记（自动带时间戳，保留最近 50 条）。"""
    from datetime import datetime
    notes = storage.load_notes(body.sid)
    notes.append({"t": datetime.now().strftime("%Y-%m-%d %H:%M"),
                  "text": body.text.strip()[:500]})
    storage.save_notes(body.sid, notes)
    return {"ok": True, "n": len(storage.load_notes(body.sid))}


@app.get("/api/report/export")
def report_export(sid: str, fmt: str = "md"):
    """📤 学习报告多格式导出：md / html / pdf / docx。"""
    from scholarpulse.core.report_exporter import export_report
    r = export_report(sid, fmt)
    if "error" in r:
        raise HTTPException(400, r["error"])
    return FileResponse(r["path"], filename=os.path.basename(r["path"]))


# ---------- 创意内容生成工坊（短剧大纲→扩写→润色→双主持→后期→多格式+视频） ----------
class CreativeIn(BaseModel):
    topic: str = ""
    genre: str = "校园短剧"
    brief: str = ""
    outline: str = ""
    script: str = ""


@app.post("/api/creative/outline")
def creative_outline(body: CreativeIn):
    """🎬 第1段：短剧大纲（4 幕起承转合，吸收自定义要求）。"""
    from scholarpulse.core.creative_studio import gen_outline
    return gen_outline(body.topic, body.genre, body.brief)


@app.post("/api/creative/expand")
def creative_expand(body: CreativeIn):
    """🎬 第2段：大纲 → 完整剧本（遵守自定义要求）。"""
    from scholarpulse.core.creative_studio import expand_script
    return expand_script(body.topic, body.outline, body.brief)


@app.post("/api/creative/polish")
def creative_polish(body: CreativeIn):
    """🎬 第3段：剧本润色。"""
    from scholarpulse.core.creative_studio import polish_script
    return polish_script(body.topic, body.script, body.brief)


@app.post("/api/creative/host")
def creative_host(body: CreativeIn):
    """🎬 第4段：双主持口播脚本（甲/乙逐行 + TTS 逐句）。"""
    from scholarpulse.core.creative_studio import dual_host_script
    return dual_host_script(body.topic, body.script, body.brief)


@app.post("/api/creative/post")
def creative_post(body: CreativeIn):
    """🎬 第5段：后期制作清单（分镜/BGM/字幕/转场）。"""
    from scholarpulse.core.creative_studio import post_production
    return post_production(body.topic, body.script, body.brief)


class CreativeExportIn(BaseModel):
    title: str = "校园短剧"
    outline: str = ""
    script: str = ""
    host: str = ""
    post: str = ""
    fmt: str = "md"


@app.post("/api/creative/export")
def creative_export(body: CreativeExportIn):
    """📤 创意脚本包多格式导出：md / html / txt。"""
    from scholarpulse.core.creative_studio import export_creative
    r = export_creative(body.title, body.outline, body.script, body.host, body.post, body.fmt)
    if "error" in r:
        raise HTTPException(400, r["error"])
    return FileResponse(r["path"], filename=os.path.basename(r["path"]))


class TtsRowIn(BaseModel):
    text: str
    key: str = ""            # 角色 key：哈希 → 固定语速声线
    offset_ms: int = 0       # 在视频时间线上的起点
    max_ms: int = 6000       # 分配到的时长（超长自动提速/截断）


class TtsIn(BaseModel):
    total_ms: int = 300000
    rows: list[TtsRowIn] = []


@app.post("/api/creative/tts")
def creative_tts(body: TtsIn):
    """🎙 短剧角色离线配音：SAPI5 逐行合成 → 按时间线混合为单条 WAV（引擎不可用返回静音轨）。"""
    from fastapi import Response
    from scholarpulse.core.tts_speaker import render_speech
    rows = [r.model_dump() for r in body.rows[:400]]
    wav = render_speech(rows, max(1000, min(body.total_ms, 900000)))
    return Response(content=wav, media_type="audio/wav")



@app.get("/api/insight/half-month")
def insight_half_month(sid: str):
    """📅 半月报告官（ReAct）：15天成长报告。"""
    from scholarpulse.agents.study_insight_agents import run_half_month_reporter
    return run_half_month_reporter(sid)


@app.get("/api/insight/resource-allocate")
def insight_resource_allocate(subject: str = "all"):
    """🏫 教育资源公平分配智能体（ReAct）：学情需求×师资供给→匹配优化→公平审计→落盘方案。"""
    from scholarpulse.agents.resource_allocator_agent import run_resource_allocator
    return run_resource_allocator(subject)


@app.get("/api/insight/learning-plan")
def insight_learning_plan(sid: str):
    """🧭 个性化学习规划师（ReAct）：读画像→规划路径→生成练习→排复习日历。"""
    from scholarpulse.agents.learning_planner_agent import run_learning_planner
    return run_learning_planner(sid)


class StudyAIIn(BaseModel):
    action: str          # tutor / similar / followup / glossary / formula / mnemonic / flashcards
    sid: str = "s001"
    concept: str = ""    # 概念名（tutor/similar/mnemonic/followup）
    message: str = ""    # followup：学生本轮作答
    turn: int = 0


@app.post("/api/study/ai")
def study_ai(body: StudyAIIn):
    """学生端 AI 学习功能统一入口（全部真 LLM）：AI 讲师 / 相似变式题 / 苏格拉底追问 /
    术语表 / 公式本 / 记忆口诀 / 闪卡。数据上下文来自该生真实掌握度。"""
    from scholarpulse.mastery import avg_mastery, weak_concepts
    from scholarpulse.llm import get_llm
    llm = get_llm()
    concepts = storage.load_concepts()
    cmap = {c.name: c for c in concepts}
    mastery = storage.load_mastery(body.sid)
    student = storage.load_students().get(body.sid)
    weak = [c.name for c in weak_concepts(mastery, concepts)[:5]]
    tier = student.tier if student else "foundation"

    def _ctx(name: str) -> str:
        c = cmap.get(name)
        if not c:
            raise HTTPException(404, f"概念「{name}」不存在")
        p = mastery[c.cid].p if c.cid in mastery else None
        return (f"概念:{c.name} 简介:{c.summary} 该生掌握度:{'未学' if p is None else round(p, 2)}"
                f" 学生层级:{tier} 全局薄弱点:{'、'.join(weak) or '无'}")

    def _items(d: dict, required: str = "") -> list:
        """键名兜底：模型可能返回 items/terms/cards 等任意键，取第一个列表值；
        再过滤缺关键字的残缺项（max_tokens 截断可能产生半条记录）。"""
        if isinstance(d, list):
            arr = d
        elif isinstance(d, dict):
            arr = next((v for v in d.values() if isinstance(v, list)), [])
        else:
            arr = []
        return [x for x in arr if isinstance(x, dict) and (not required or x.get(required))]

    if body.action == "tutor":
        sys_ = ("你是 AI 讲师，面向大学生讲解一个考点。要求：①先给一句生活化类比；"
                "②再讲核心机制（不超过 5 句）；③结合学生当前掌握度给下一步动作。"
                "全文 150 字以内，口语化。")
        return {"text": llm.chat(sys_, _ctx(body.concept), max_tokens=400)}
    if body.action == "similar":
        sys_ = ('基于给出的考点与学生掌握度，出 3 道相似但不同角度的变式题。'
                '只输出 JSON：{"items":[{"q":"题干","a":"参考答案"}]}，每题 a 不超过 40 字。')
        return {"items": _items(llm.chat_json(sys_, _ctx(body.concept)), "q")}
    if body.action == "followup":
        sys_ = ("你在用苏格拉底式追问帮学生检验对考点的理解：每次只问一个 "
                "启发式问题（不给答案）；若学生回答暴露误区，先指出矛盾再追问。"
                "50 字以内。这是第 " + str(body.turn + 1) + " 轮。")
        user = _ctx(body.concept) + f"\n学生本轮回答:{body.message or '（刚开始，请提出第一个追问）'}"
        return {"text": llm.chat(sys_, user, temperature=0.5, max_tokens=200)}
    if body.action == "glossary":
        sys_ = ('把给出的考点列表整理成术语表。只输出 JSON：'
                '{"items":[{"term":"术语","explain":"一句话解释(≤30字)"}]}。')
        return {"items": _items(llm.chat_json(sys_, "考点:" + "、".join(c.name for c in concepts), max_tokens=900), "term")}
    if body.action == "formula":
        sys_ = ('从考点列表中挑出涉及公式/规则/判定的条目整理成公式本；'
                '没有公式条目则给核心判定规则。只输出 JSON：'
                '{"items":[{"name":"名称","formula":"公式或规则","note":"使用场景(≤25字)"}]}，最多 8 条。')
        return {"items": _items(llm.chat_json(sys_, "考点:" + "、".join(c.name for c in concepts), max_tokens=900), "name")}
    if body.action == "mnemonic":
        sys_ = ("为该考点编一个记忆口诀或联想记忆（结合生活场景），"
                "先给口诀（≤20字），再一句解释为什么好记。60 字以内。")
        return {"text": llm.chat(sys_, _ctx(body.concept), temperature=0.8, max_tokens=200)}
    if body.action == "flashcards":
        sys_ = ('把考点列表做成闪卡。只输出 JSON：'
                '{"items":[{"front":"正面问题","back":"背面答案"}]}，最多 8 张，back ≤30 字。')
        return {"items": _items(llm.chat_json(sys_, "考点(附简介):" + ";".join(f"{c.name}({c.summary[:30]})" for c in concepts), max_tokens=900), "front")}
    if body.action == "interrogate":
        # 元认知审讯官：message = "题干||参考答案"，还原学生答错时的决策链
        q, _, ref = (body.message or "").partition("||")
        sys_ = ("你是元认知审讯官。学生做错了一道题，你的任务不是给答案，"
                "而是用 3 个递进追问还原他当时的决策过程：①审题层——他可能把题干理解成了什么？"
                "②知识层——他调用了哪个（可能错误的）知识点？③信念层——为什么这个错误对他来说『合理』？"
                "最后一行给干预建议。共 4 行，每行 ≤30 字，行首用 ①②③④。")
        return {"text": llm.chat(sys_, f"题目:{q} 参考答案:{ref} 考点:{body.concept}",
                                 temperature=0.5, max_tokens=320)}
    raise HTTPException(400, f"未知 action: {body.action}")


class BuddyIn(BaseModel):
    sid: str


def _match_buddy(sid: str, students: dict, concepts: list, mastery: dict,
                 diversify: bool = False, pinned_sid: str | None = None) -> dict:
    """按掌握度向量匹配互补搭子：他强我弱的考点数（互补度）×2 + 余弦相似度。
    diversify=True 时从 Top-5 中随机选且避开上一次的搭子（每次匹配遇到不同人）；
    pinned_sid 指定时返回该搭子（Boss战/天梯/聊天室锁定同一搭子）。"""
    import math
    cids = [c.cid for c in concepts]

    def vec(m: dict) -> list:
        return [m[c].p if c in m else 0.0 for c in cids]

    def cos(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    v_me = vec(mastery)
    weak_me = {c for c in cids if (mastery[c].p if c in mastery else 0.0) < 0.45}
    ranked = []
    for sid2, s2 in students.items():
        if sid2 == sid:
            continue
        m2 = storage.load_mastery(sid2)
        v2 = vec(m2)
        weak2 = {c for c in cids if (m2[c].p if c in m2 else 0.0) < 0.45}
        comp = len(weak_me & (set(cids) - weak2))
        sim = cos(v_me, v2)
        ranked.append({"score": comp * 2 + sim, "sid": sid2, "student": s2, "mastery": m2,
                       "sim": round(sim, 3), "comp": comp,
                       "weak": weak2, "weak_me": weak_me, "cids": cids})
    ranked.sort(key=lambda c: -c["score"])
    if pinned_sid and pinned_sid in students:
        best = next(c for c in ranked if c["sid"] == pinned_sid)
    elif diversify:
        import random
        pf = f"data/buddy_current_{sid}.json"
        last = json.load(open(pf, encoding="utf-8")).get("sid") if os.path.exists(pf) else None
        pool = [c for c in ranked[:5] if c["sid"] != last] or ranked[:5]
        best = random.choice(pool)
        json.dump({"sid": best["sid"], "name": best["student"].name},
                  open(pf, "w", encoding="utf-8"), ensure_ascii=False)
    else:
        best = ranked[0]
    return best


@app.post("/api/study/buddy")
def study_buddy(body: BuddyIn):
    """学习搭子：按掌握度向量匹配互补搭子（他的强项=我的弱项优先），
    展示组队理由与共同薄弱考点（Boss 战的引子）。"""
    from scholarpulse.mastery import weak_concepts
    from scholarpulse.llm import get_llm
    students = storage.load_students()
    me = students.get(body.sid)
    if not me:
        raise HTTPException(404, f"学生 {body.sid} 不存在")
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    mastery = storage.load_mastery(body.sid)
    b = _match_buddy(body.sid, students, concepts, mastery, diversify=True)
    b_weak = weak_concepts(b["mastery"], concepts)[:2]
    my_weak = weak_concepts(mastery, concepts)[:2]
    llm = get_llm()
    tier2 = b["student"].tier
    # 师徒制关系判定：均掌握度差 >0.05 → 强者为师，否则势均力敌互为对手
    avg_me = sum(x.p for x in mastery.values()) / max(len(mastery), 1)
    avg_b = sum(x.p for x in b["mastery"].values()) / max(len(b["mastery"]), 1)
    if avg_b - avg_me > 0.05:
        rel, rel_txt = "mentor", f"🎓 他当你导师（班均 {avg_b:.2f} vs 你的 {avg_me:.2f}）"
    elif avg_me - avg_b > 0.05:
        rel, rel_txt = "mentee", f"🌱 你当他导师（班均 {avg_me:.2f} vs 他的 {avg_b:.2f}）"
    else:
        rel, rel_txt = "rival", f"⚔️ 势均力敌，互为对手（班均 {avg_b:.2f} vs {avg_me:.2f}）"
    # 共同薄弱考点 = 双方都弱的（Boss 战的猎物）
    both_weak = sorted(b["weak_me"] & b["weak"],
                       key=lambda c: (mastery[c].p if c in mastery else 0.0))
    both_names = [name_of.get(c, c) for c in both_weak][:3]
    narrative = llm.chat(
        "你是学习搭子匹配官，40 字内给出组队理由与一起刷题的建议（结合双方数据）。",
        f"我:{me.name} 层级{me.tier} 均分{sum(x.p for x in mastery.values())/max(len(mastery),1):.2f} "
        f"薄弱:{'、'.join(c.name for c in my_weak) or '无'}\n"
        f"搭子:{b['student'].name} 层级{tier2} 相似度{b['sim']} 互补考点{b['comp']}个 "
        f"薄弱:{'、'.join(c.name for c in b_weak) or '无'} "
        f"共同薄弱:{'、'.join(both_names) or '无'}",
        temperature=0.6, max_tokens=150)
    return {
        "me": {"name": me.name, "tier": me.tier},
        "buddy": {"sid": b["sid"], "name": b["student"].name, "tier": tier2,
                  "sim": b["sim"], "comp": b["comp"],
                  "weak": [c.name for c in b_weak]},
        "relation": rel, "relation_txt": rel_txt,
        "both_weak": both_names,
        "narrative": narrative,
    }


class BossIn(BaseModel):
    sid: str
    answer: str = ""     # 为空=召唤 Boss 出题；非空=提交我的作答进入裁决


@app.post("/api/study/buddy/battle")
def buddy_battle(body: BossIn):
    """组队讨伐知识 Boss（三智能体协作攻防）：
    阶段1 —— 取双方共同最薄弱考点，Boss 智能体（嘲讽风格）出关底综合题；
    阶段2 —— 我作答 + 搭子智能体按画像补充作答（共同弱点，他也会卡壳），
             Boss 裁决两人答案合并后是否覆盖全部要点，输出讨伐战报。"""
    from scholarpulse.llm import get_llm
    llm = get_llm()
    students = storage.load_students()
    concepts = storage.load_concepts()
    name_of = {c.cid: c.name for c in concepts}
    mastery = storage.load_mastery(body.sid)
    b = _match_buddy(body.sid, students, concepts, mastery, pinned_sid=_current_buddy_sid(body.sid))
    me = students[body.sid]

    if not body.answer:
        # 阶段1：召唤 Boss
        both = b["weak_me"] & b["weak"]
        if not both:
            raise HTTPException(400, "你俩没有共同薄弱考点，无法召唤 Boss（这是好事！）")
        boss_cid = min(both, key=lambda c: (mastery[c].p if c in mastery else 0.0))
        boss_c = name_of.get(boss_cid, boss_cid)
        summary = next((c.summary for c in concepts if c.cid == boss_cid), "")
        q = llm.chat_json(
            '你是「知识 Boss」——一个考点的终极守门员，说话带着一点傲慢的挑衅。'
            '围绕该考点出一道关底综合题：需要两步推理或结合场景应用，并把题干里埋一个常见误区。'
            '只输出 JSON：{"q":"题干(含场景，≤60字)","ref":"要点式参考答案，要点用分号分隔","taunt":"Boss 的开场嘲讽(≤25字)"}',
            f"考点:{boss_c} 简介:{summary[:80]} 这两个学生都还没掌握它",
            max_tokens=350)
        json.dump({"cid": boss_cid, "boss_c": boss_c, "q": q.get("q", ""), "ref": q.get("ref", ""),
                   "buddy_sid": b["sid"], "buddy_name": b["student"].name,
                   "buddy_tier": b["student"].tier},
                  open(f"data/pending_boss_{body.sid}.json", "w", encoding="utf-8"), ensure_ascii=False)
        return {"stage": "question", "boss_c": boss_c, "question": q.get("q", ""),
                "taunt": q.get("taunt", ""), "buddy_name": b["student"].name}
    # 阶段2：协作作答 + Boss 裁决
    pfile = f"data/pending_boss_{body.sid}.json"
    if not os.path.exists(pfile):
        raise HTTPException(400, "请先召唤 Boss")
    p = json.load(open(pfile, encoding="utf-8"))
    ans_buddy = llm.chat(
        f"你是学生{p['buddy_name']}（层级{p['buddy_tier']}），这个考点你也没掌握好，"
        "尽力答，可能只答对一部分，也可能暴露典型误区。35 字内直接作答。",
        f"题目:{p['q']} 参考要点:{p['ref']} 该考点你的掌握情况:薄弱",
        temperature=0.7, max_tokens=140)
    verdict = llm.chat_json(
        '你是知识 Boss，正在验收两名学生的联手一击。把两人答案的要点合并起来与参考要点对照：'
        '合并覆盖了哪些要点、漏了哪些。只输出 JSON：'
        '{"beaten":true或false,"found":["答对的要点"],"missed":["漏掉的要点"],"report":"讨伐战报(≤50字，若击破给出认可，若失败给一句挑衅)"}',
        f"题目:{p['q']} 参考要点:{p['ref']}\n我的答案:{body.answer}\n搭子的答案:{ans_buddy}",
        max_tokens=350)
    beaten = bool(verdict.get("beaten"))
    if beaten:
        os.remove(pfile)   # 击破后 Boss 消失，可再召唤
    return {"stage": "verdict", "boss_c": p["boss_c"], "buddy_name": p["buddy_name"],
            "buddy_answer": ans_buddy,
            "beaten": beaten, "found": verdict.get("found", []),
            "missed": verdict.get("missed", []), "report": verdict.get("report", "")}


def _current_buddy_sid(sid: str) -> str | None:
    """取当前锁定搭子（匹配时写入），保证玩法会话中搭子稳定。"""
    pf = f"data/buddy_current_{sid}.json"
    return json.load(open(pf, encoding="utf-8")).get("sid") if os.path.exists(pf) else None


class LadderIn(BaseModel):
    sid: str
    action: str            # start / play / board
    answer: str = ""


@app.post("/api/study/ladder")
def ladder(body: LadderIn):
    """搭子天梯赛：3 局制对战——每局双方同答一题，AI 裁判各打分（0-10），
    比分领先则下一题变难；3 局后终局判定并累计积分榜（持久化）。"""
    from scholarpulse.llm import get_llm
    from scholarpulse.mastery import weak_concepts
    llm = get_llm()
    students = storage.load_students()
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(body.sid)
    if body.action == "board":
        # 全班积分榜：汇总所有玩过天梯赛的战绩文件
        import glob
        rows = []
        for f in glob.glob("data/ladder_*.json"):
            d = json.load(open(f, encoding="utf-8"))
            sid2 = os.path.basename(f)[len("ladder_"):-len(".json")]
            s2 = students.get(sid2)
            rows.append({"sid": sid2, "name": s2.name if s2 else sid2,
                         "w": d.get("w", 0), "l": d.get("l", 0), "d": d.get("d", 0),
                         "pts": d.get("w", 0) * 3 + d.get("d", 0)})
        rows.sort(key=lambda r: (-r["pts"], r["name"]))
        my_rank = next((i + 1 for i, r in enumerate(rows) if r["sid"] == body.sid), None)
        return {"board": rows[:8], "my_rank": my_rank}
    b = _match_buddy(body.sid, students, concepts, mastery, pinned_sid=_current_buddy_sid(body.sid))
    pfile = f"data/pending_ladder_{body.sid}.json"
    if body.action == "start":
        q = llm.chat_json(
            '你是对战出题官，出 1 道简答题（30-60 字题干）。只输出 JSON：{"q":"题干","ref":"要点式参考答案，要点用分号分隔"}',
            f"课程考点:{'、'.join(c.name for c in concepts[:6])}", max_tokens=280)
        st = {"round": 1, "my": [], "bd": [], "q": q.get("q", ""), "ref": q.get("ref", ""),
              "buddy_sid": b["sid"], "buddy_name": b["student"].name,
              "buddy_tier": b["student"].tier, "b_weak": [c.name for c in weak_concepts(b["mastery"], concepts)[:2]]}
        json.dump(st, open(pfile, "w", encoding="utf-8"), ensure_ascii=False)
        return {"round": 1, "total": 3, "question": st["q"],
                "buddy_name": st["buddy_name"], "score": {"me": 0, "buddy": 0}}
    # action == play
    if not os.path.exists(pfile):
        raise HTTPException(400, "请先开始天梯赛")
    st = json.load(open(pfile, encoding="utf-8"))
    ans_buddy = llm.chat(
        f"你是学生{st['buddy_name']}（层级{st['buddy_tier']}），与同学对战答题。"
        f"你的薄弱考点:{'、'.join(st['b_weak']) or '无'}——若题目涉及，答案会含糊或漏要点。40 字内直接作答。",
        f"题目:{st['q']} 参考要点:{st['ref']}", temperature=0.7, max_tokens=140)
    judge = llm.chat_json(
        '你是对战裁判。仔细比对两个答案与参考要点的覆盖度，各给出一个 0 到 10 之间的数值分'
        '（可以是 7.5 这样的小数；要点覆盖越全分越高，答非所问才给 0-2 分）。'
        '只输出 JSON：{"my_score":<数值>,"buddy_score":<数值>,"comment":"≤30字点评双方"}',
        f"题目:{st['q']} 参考要点:{st['ref']}\n{students[body.sid].name}的答案:{body.answer}\n{st['buddy_name']}的答案:{ans_buddy}",
        max_tokens=220)
    ms, bs = float(judge.get("my_score", 0)), float(judge.get("buddy_score", 0))
    st["my"].append(ms); st["bd"].append(bs)
    total_round, done = 3, st["round"] >= 3
    result = {"round": st["round"], "total": total_round,
              "buddy_name": st["buddy_name"], "buddy_answer": ans_buddy,
              "my_score": ms, "buddy_score": bs, "comment": judge.get("comment", ""),
              "score": {"me": round(sum(st["my"]), 1), "buddy": round(sum(st["bd"]), 1)}}
    if done:
        my_t, bd_t = sum(st["my"]), sum(st["bd"])
        outcome = "w" if my_t > bd_t else ("l" if my_t < bd_t else "d")
        lad = storage.load_ladder(body.sid)
        lad[outcome] += 1
        lad["history"].append({"vs": st["buddy_name"], "me": round(my_t, 1),
                               "bd": round(bd_t, 1), "r": outcome,
                               "ts": storage.now_iso()})
        lad["history"] = lad["history"][-10:]
        storage.save_ladder(body.sid, lad)
        os.remove(pfile)
        result.update({"final": True, "outcome": outcome,
                       "verdict": ("🏆 你赢了！" if outcome == "w" else
                                   "💪 输了，下次找回来" if outcome == "l" else "🤝 平分秋色")})
    else:
        st["round"] += 1
        diff = "更难（你暂时领先，出题官加大力度）" if sum(st["my"]) > sum(st["bd"]) else "稍简单（给你追分机会）"
        q2 = llm.chat_json(
            f'你是对战出题官，出 1 道简答题，难度比上一题{diff}。只输出 JSON：{{"q":"题干","ref":"要点式参考答案，分号分隔"}}',
            f"课程考点:{'、'.join(c.name for c in concepts[:6])} 上一题:{st['q']}", max_tokens=280)
        st["q"], st["ref"] = q2.get("q", ""), q2.get("ref", "")
        json.dump(st, open(pfile, "w", encoding="utf-8"), ensure_ascii=False)
        result.update({"final": False, "question": st["q"], "round": st["round"]})
    return result


class BuddyChatIn(BaseModel):
    sid: str
    message: str = ""
    reset: bool = False


@app.post("/api/study/buddy/chat")
def buddy_chat(body: BuddyChatIn):
    """搭子讨论聊天室：与 AI 扮演的搭子同学多轮讨论（会话记忆持久化）。
    搭子有真实画像：强项答得利索，薄弱点会坦言不会，并提议一起查资料/问 AI 讲师。"""
    from scholarpulse.mastery import weak_concepts
    from scholarpulse.llm import get_llm
    hist = storage.load_buddy_chat(body.sid)
    if body.reset:
        storage.save_buddy_chat(body.sid, [])
        return {"history": []}
    students = storage.load_students()
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(body.sid)
    b = _match_buddy(body.sid, students, concepts, mastery, pinned_sid=_current_buddy_sid(body.sid))
    me = students[body.sid]
    b_strong = [c.name for c in concepts
                if (b["mastery"][c.cid].p if c.cid in b["mastery"] else 0) >= 0.6][:3]
    b_weak = [c.name for c in weak_concepts(b["mastery"], concepts)[:3]]
    if not body.message:
        return {"history": hist, "buddy_name": b["student"].name}
    llm = get_llm()
    sys_ = (f"你是学生{b['student'].name}（层级{b['student'].tier}），正在 App 里和同学{me.name}讨论课程问题。"
            f"你的强项:{'、'.join(b_strong) or '无'}，薄弱:{'、'.join(b_weak) or '无'}。"
            "用同学之间的口语口吻回答（不是老师），≤80 字；"
            "遇到你的薄弱点要坦诚说『这我也没整明白』，并提议一起查资料或去问 AI 讲师；"
            "禁止说自己是 AI 或助手。")
    hist.append({"role": "me", "text": body.message})
    ctx = "\n".join(f"{'我' if m['role'] == 'me' else b['student'].name}:{m['text']}" for m in hist[-6:])
    reply = llm.chat(sys_, f"最近对话：\n{ctx}\n\n请以{b['student'].name}的身份回复最后一条消息。",
                     temperature=0.7, max_tokens=200)
    hist.append({"role": "buddy", "text": reply})
    storage.save_buddy_chat(body.sid, hist)
    return {"history": hist[-12:], "reply": reply, "buddy_name": b["student"].name}


TABLE_PERSONAS = ["喜欢举生活例子", "较真爱抬杠但讲道理", "反应快话多", "慢热但一针见血",
                  "自嘲式幽默", "爱总结套路", "爱用类比解释", "喜欢反问引导别人"]


class TableIn(BaseModel):
    sid: str
    action: str            # open / say / reset
    message: str = ""


@app.post("/api/study/buddy/roundtable")
def roundtable(body: TableIn):
    """学习圆桌（三智能体群聊动力学）：我 + 当前搭子 + 随机第三位同学。
    开桌=随机选人+每人随机性格人设+按三方薄弱点生成今日议题；
    发言后群聊动力学决定谁回应（70% 两人都回，30% 单人回），发言顺序随机，
    后发言者必须对先发言者做出明确互动（引用/反驳/补充并点名）——观点碰撞。"""
    import random as _rd
    from scholarpulse.mastery import weak_concepts
    from scholarpulse.llm import get_llm
    tf = f"data/roundtable_{body.sid}.json"
    if body.action == "reset":
        if os.path.exists(tf):
            os.remove(tf)
        return {"cleared": True}
    students = storage.load_students()
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(body.sid)
    me = students[body.sid]
    llm = get_llm()
    name_of = {c.cid: c.name for c in concepts}

    def _prof(sid2: str, s2, m2: dict) -> dict:
        st = [c.name for c in concepts if (m2[c.cid].p if c.cid in m2 else 0) >= 0.6][:3]
        wk = [c.name for c in weak_concepts(m2, concepts)[:3]]
        return {"sid": sid2, "name": s2.name, "tier": s2.tier,
                "strong": "、".join(st) or "无", "weak": "、".join(wk) or "无",
                "persona": _rd.choice(TABLE_PERSONAS)}

    if body.action == "open":
        b = _match_buddy(body.sid, students, concepts, mastery, pinned_sid=_current_buddy_sid(body.sid))
        third_sid = _rd.choice([s for s in students if s not in (body.sid, b["sid"])])
        third_m = storage.load_mastery(third_sid)
        members = [_prof(body.sid, me, mastery), _prof(b["sid"], b["student"], b["mastery"]),
                   _prof(third_sid, students[third_sid], third_m)]
        weak_sets = [set(c.cid for c in weak_concepts(m2, concepts))
                     for m2 in (mastery, b["mastery"], third_m)]
        common = (weak_sets[0] & weak_sets[1] & weak_sets[2]) or \
                 ((weak_sets[0] & weak_sets[1]) | (weak_sets[0] & weak_sets[2]) |
                  (weak_sets[1] & weak_sets[2])) or weak_sets[0]
        focus = "、".join(sorted(name_of.get(c, c) for c in common)[:2]) or "最近的课程内容"
        topic = llm.chat("你是圆桌主持人，用一句有钩子的议题开场（≤35字，带一个具体疑问），议题围绕：",
                         focus, temperature=0.8, max_tokens=80)
        table = {"members": members, "topic": topic, "history": []}
        json.dump(table, open(tf, "w", encoding="utf-8"), ensure_ascii=False)
        return {"members": members, "topic": topic, "history": []}
    if not os.path.exists(tf):
        raise HTTPException(400, "请先开桌")
    table = json.load(open(tf, encoding="utf-8"))
    if body.action == "analyze":
        """深度分析：LLM 通读全量讨论（含归档）+ 三人画像 → 观点地图
        （共识 / 分歧对垒 / 暴露的盲区 / 下一步建议），落盘供重复查看。"""
        all_h = table.get("archive", []) + table["history"]
        if len(all_h) < 2:
            raise HTTPException(400, "讨论还太少，先聊两轮再分析")
        lines = "\n".join(f"{h['name']}:{h['text']}" for h in all_h)
        mems = "\n".join(f"{m['name']}（性格:{m['persona']}；强项:{m['strong']}；薄弱:{m['weak']}）"
                         for m in table["members"])
        a = llm.chat_json(
            '你是学习圆桌的深度分析师。通读完整讨论并对照三人画像，输出结构化分析。只输出 JSON：'
            '{"consensus":["已达成的共识，每条一句话"],"disputes":[{"who":"谁","point":"其观点","vs":"与谁的什么观点分歧"}],'
            '"blind_spots":[{"who":"谁","gap":"他在讨论中暴露的知识盲区"}],'
            '"growth":"给全组下一步学习建议，≤40字"}。'
            'disputes 没有就说没有；盲区要基于发言内容推断，不臆造。',
            f"成员画像：\n{mems}\n今日议题：{table['topic']}\n完整讨论：\n{lines}",
            temperature=0.4, max_tokens=600)
        table["analysis"] = a
        json.dump(table, open(tf, "w", encoding="utf-8"), ensure_ascii=False)
        return {"analysis": a}
    # action == say
    others = [m for m in table["members"] if m["sid"] != body.sid]
    table["history"].append({"name": me.name, "role": "me", "text": body.message})
    # 群聊长期记忆：发言超量时把早期对话压缩为「讨论脉络」常驻后续 prompt——前后轮永不失联
    if len(table["history"]) > 10:
        old = table["history"][:-6]
        table["archive"] = table.get("archive", []) + old
        table["history"] = table["history"][-6:]
        if not table.get("summary"):
            old_txt = "\n".join(f"{h['name']}:{h['text']}" for h in old)
            table["summary"] = llm.chat(
                "你是圆桌记录员，50 字内总结此前的讨论脉络（谁主张了什么、形成了哪些共识、留了什么疑问）。",
                old_txt, temperature=0.5, max_tokens=130)
    speakers = list(others) if _rd.random() < 0.7 else [_rd.choice(others)]
    _rd.shuffle(speakers)
    replies = []
    for i, sp in enumerate(speakers):
        ctx = "\n".join(f"{h['name']}:{h['text']}" for h in table["history"][-8:])
        if i == 0:
            rule = "自然接话。"
        else:
            rule = (f"{speakers[0]['name']} 刚刚发言了（见上文最后）——你必须明确回应他："
                    "要么补充、要么反驳、要么引用他的话再给自己的看法，要点名互动。")
        sys_ = (f"你是学生{sp['name']}（层级{sp['tier']}），性格：{sp['persona']}。"
                f"正在班级学习圆桌里和同学讨论。你的强项:{sp['strong']}，薄弱:{sp['weak']}——"
                "谈到你的薄弱点可以坦诚不会并向大家求教。≤60字，同学口语，禁止说自己是AI。"
                f"{rule}")
        prefix = f"此前讨论脉络：{table.get('summary', '')}\n" if table.get("summary") else ""
        reply = llm.chat(sys_, f"今日议题：{table['topic']}\n{prefix}最近对话：\n{ctx}",
                         temperature=0.8, max_tokens=170)
        table["history"].append({"name": sp["name"], "role": "mate", "text": reply})
        replies.append({"name": sp["name"], "persona": sp["persona"], "text": reply})
    json.dump(table, open(tf, "w", encoding="utf-8"), ensure_ascii=False)
    return {"replies": replies, "history": table["history"][-20:], "topic": table["topic"]}


@app.post("/api/teacher/consult")
def teacher_consult(body: ConsultIn):
    """教研会诊：一名掉队学生 → 本班导师/年级教研组长/学习科学专家三智能体独立诊断
    → 教学主席共识方案（任一角色失败降级为规则视图，会诊永不空转）。"""
    from scholarpulse.graph import consult
    if not body.sid.strip():
        raise HTTPException(400, "学号不能为空")
    try:
        return consult.run_consult(body.sid.strip())
    except ValueError as e:
        raise HTTPException(404, str(e))


class TeacherAskIn(BaseModel):
    message: str
    session_id: str = ""


_teacher_sessions: dict[str, list] = {}   # session_id -> [(问, 答)]，内存会话记忆


@app.post("/api/teacher/assistant")
def teacher_assistant(body: TeacherAskIn):
    """教师端教学助手：复用 ReAct 副官引擎（17 工具池）+ 教师人设 + 三步诊断 SOP，
    自主多步调用工具后给出教学建议；按 session_id 记忆最近 6 轮对话支持连续追问。"""
    import uuid
    from scholarpulse.graph.chief import ask_chief
    sid = body.session_id or uuid.uuid4().hex
    hist = _teacher_sessions.get(sid, [])[-6:]
    r = ask_chief(body.message, role="teacher", history=hist)
    _teacher_sessions[sid] = (hist + [(body.message, r.get("answer", ""))])[-6:]
    r["session_id"] = sid
    return r


# ---------- 学区级多智能体治理（L0-L4）：学区总监驾驶舱 /officer ----------
class SeedIn(BaseModel):
    seed: int = 42
    sim_days: int | None = None


class SimulateIn(BaseModel):
    days: int = 30
    seed: int = 42


@app.get("/officer")
def officer_page():
    return FileResponse(os.path.join(STATIC, "officer.html"))


@app.post("/api/society/seed")
def society_seed(body: SeedIn):
    """L0 人口合成：80 教师 + 350 合成学生 + 社交图。sim_days 提供时顺手跑 ABM。"""
    from scholarpulse.society.population import seed_society
    info = seed_society(body.seed)
    out = {"teachers": info["teachers"], "students": info["students"],
           "edges": info["edges"], "groups": info["groups"], "concepts": info["concepts"]}
    if body.sim_days:
        out["simulate"] = society_simulate(SimulateIn(days=body.sim_days, seed=body.seed))
    return out


@app.post("/api/society/simulate")
def society_simulate(body: SimulateIn):
    """L1 ABM：跑 days 个虚拟天并批量落盘 700 个终态文件。"""
    from scholarpulse.society import abm
    return abm.run_days(days=body.days, seed=body.seed)


@app.get("/api/society/overview")
def society_overview():
    """驾驶舱 KPI + 80 班横条图数据（真实学生与合成人口统一视图）。"""
    students = storage.load_students()
    teachers = storage.load_teachers()
    tier_dist = {"foundation": 0, "consolidation": 0, "extension": 0}
    roster: dict[str, list[str]] = {}
    for sid, s in students.items():
        roster.setdefault(s.teacher_id, []).append(sid)
        if s.tier in tier_dist:
            tier_dist[s.tier] += 1
    classes, all_avg = [], []
    for tid, t in sorted(teachers.items()):
        avgs, at_risk = [], 0
        for sid in roster.get(tid, []):
            m = storage.load_mastery(sid)
            if not m:
                continue
            a = M.avg_mastery(m)
            avgs.append(a)
            if a < 0.4:
                at_risk += 1
        avg = sum(avgs) / len(avgs) if avgs else 0.0
        classes.append({"tid": tid, "name": t.name, "n": len(avgs),
                        "avg": round(avg, 4), "at_risk": at_risk})
        all_avg.extend(avgs)
    overall = round(sum(all_avg) / len(all_avg), 4) if all_avg else 0.0
    return {"n_teachers": len(teachers), "n_students": len(students),
            "avg_mastery": overall, "tier_dist": tier_dist, "classes": classes,
            "risk_classes": sum(1 for c in classes if c["avg"] < 0.5)}


class GovIn(BaseModel):
    top_k: int = 5


@app.post("/api/governance/analyze")
def governance_analyze(body: GovIn):
    """L2 层级治理：Send 扇出 80 教师画像 → Top-K LLM 深析 → 总监归约（约 30-60s）。"""
    from scholarpulse.graph import hierarchy
    return hierarchy.run_governance(top_k=max(1, body.top_k))


@app.get("/api/governance/result")
def governance_result():
    rep = storage.load_governance()
    return rep.model_dump() if rep else {}


@app.get("/api/epidemic/dashboard")
def epidemic_dashboard():
    """L3 误区疫情仪表盘：SIR 曲线 + 逐概念 R0 + 超级传播者。"""
    from scholarpulse.society import epidemic
    data = epidemic.dashboard()
    cmap = {c.cid: c.name for c in storage.load_concepts()}
    for r in data.get("r0_list", []):
        r["name"] = cmap.get(r["cid"], r["cid"])
    return data


class InterveneIn(BaseModel):
    cid: str
    action: str = "regroup"            # regroup 拆组 / vaccine 疫苗式复习
    target_sids: list[str] | None = None


@app.post("/api/epidemic/intervene")
def epidemic_intervene(body: InterveneIn):
    """反事实干预回测：同 seed 重放传播动力学，量化干预效果。"""
    from scholarpulse.society import epidemic
    if body.action not in ("regroup", "vaccine"):
        raise HTTPException(400, "action 必须是 regroup 或 vaccine")
    return epidemic.design_intervention(body.cid, body.action, body.target_sids)


class PolicyIn(BaseModel):
    policy_text: str
    days: int = 30


@app.post("/api/policy/sandbox")
def policy_sandbox_api(body: PolicyIn):
    """L4 政策沙盘：解析→RCT分组→双世界仿真→bootstrap CI→顾问（约 10-40s）。"""
    from scholarpulse.graph import policy_sandbox
    if not body.policy_text.strip():
        raise HTTPException(400, "政策文本不能为空")
    return policy_sandbox.run_sandbox(body.policy_text.strip(), days=max(1, body.days))


class ChiefIn(BaseModel):
    message: str
    session_id: str = ""


@app.post("/api/policy/debate")
def policy_debate_api():
    """L4b 政策听证委员会：三派两轮辩论 + 主持人裁决（需先有有效沙盘结果）。"""
    from scholarpulse.graph import policy_debate
    try:
        return policy_debate.run_debate()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/policy/debate/result")
def policy_debate_result():
    from scholarpulse import storage as st
    d = st.load_debate()
    if d is None:
        raise HTTPException(404, "尚未举行听证会")
    return d.model_dump()


_chief_sessions: dict[str, list] = {}    # 管理员端副官会话记忆：session_id -> [(问, 答)]


@app.post("/api/chief/ask")
def chief_ask(body: ChiefIn):
    """总监副官：ReAct 自主智能体，多步工具调用轨迹 + 最终答复；按 session 记忆 6 轮。"""
    import uuid
    from scholarpulse.graph import chief
    if not body.message.strip():
        raise HTTPException(400, "问题不能为空")
    sid = body.session_id or uuid.uuid4().hex
    hist = _chief_sessions.get(sid, [])[-6:]
    r = chief.ask_chief(body.message.strip(), role="chief", history=hist)
    _chief_sessions[sid] = (hist + [(body.message.strip(), r.get("answer", ""))])[-6:]
    r["session_id"] = sid
    return r


@app.get("/api/district/rag_insight")
def district_rag_insight(top_k: int = 4):
    """🧠 检索增强学区洞察：把三通道混合检索算法深嵌学区治理链路。
    取误区疫情 R0 最高的考点，逐个用『特征哈希密集向量 + CSR 稀疏矩阵 BM25
    + 硅基流动语义嵌入(可选) → RRF 融合』检索学区语料库（s001 课件）取证
    课件原文，GraphRAG 扩展先修上下文，最后 LLM 汇总一份只引用证据的行动简报。"""
    from scholarpulse.society import epidemic
    from scholarpulse.core.siliconflow_embed import hybrid_search
    from scholarpulse.core.rag_retriever import expand_with_graph
    from scholarpulse.llm import get_llm
    data = epidemic.dashboard()
    cmap = {c.cid: c.name for c in storage.load_concepts()}
    top = sorted(data.get("r0_list", []), key=lambda x: -x.get("r0", 0))[:max(1, top_k)]
    if not top:
        return {"items": [], "msg": "无疫情数据：请先在驾驶舱完成 ①② 仿真"}
    concepts = storage.load_concepts()
    mastery = storage.load_mastery("s001")
    items = []
    for row in top:
        q = row.get("name") or cmap.get(row.get("cid"), str(row.get("cid")))
        out = hybrid_search("s001", q, top_k=3)
        hits = out.get("hits", [])
        items.append({"cid": row.get("cid"), "name": q,
                      "r0": round(row.get("r0", 0), 2),
                      "infected_n": row.get("infected_n"),
                      "channels": out.get("channels", 2),
                      "embed_model": out.get("embed_model"), "hits": hits,
                      "graph_context": expand_with_graph(hits, concepts, mastery)})
    quotes = "\n".join(f"[{it['name']} R0={it['r0']}] {h['text'][:80]}"
                       for it in items for h in it["hits"][:2])
    brief, actions, note = "", [], ""
    try:
        res = get_llm().chat_json(
            "你是学区总监的政策分析师。基于给定检索证据（课件原文片段，附误区传播R0）"
            "输出JSON：{\"brief\":\"≤100字学区行动简报（只引用证据中的事实，禁止编造）\","
            "\"actions\":[\"3条具体行动\"]}",
            f"检索证据：\n{quotes}", temperature=0.4, max_tokens=320)
        brief = res.get("brief", "")
        actions = [str(a) for a in (res.get("actions") or [])][:3]
    except Exception as e:
        note = f"LLM 简报不可用（{e}），仅展示检索证据"
    return {"items": items, "brief": brief, "actions": actions, "note": note}


@app.get("/api/teacher/teach_pack")
def teacher_teach_pack(tid: str, top_k: int = 3):
    """🧬 检索增强备课包：班级薄弱考点 × 全校误区疫情(R0) × 三通道混合检索。
    零 LLM——证据全部由确定性引擎产出：班级概念级薄弱聚合（确定性统计）→
    与 epidemic R0 交叉算教学优先级（班级缺口 × 全校传播力）→
    『密集哈希+CSR稀疏BM25+语义嵌入→RRF』检索课件原文 + GraphRAG 先修链。"""
    from collections import defaultdict
    from scholarpulse import storage, mastery as M
    from scholarpulse.society import epidemic
    from scholarpulse.core.siliconflow_embed import hybrid_search
    from scholarpulse.core.rag_retriever import expand_with_graph
    students = storage.load_students()
    pool = {sid: s for sid, s in students.items() if s.teacher_id == tid} or students
    msts = {sid: storage.load_mastery(sid) for sid in pool}
    # 班级概念级薄弱聚合（确定性统计，无 LLM）
    agg: dict[str, list] = defaultdict(lambda: [0.0, 0])
    for m in msts.values():
        for cid, rec in m.items():
            a = agg[cid]
            a[0] += rec.p
            a[1] += 1
    weak = sorted(((cid, s / n) for cid, (s, n) in agg.items() if n >= 3),
                  key=lambda x: x[1])[:max(1, top_k) * 2]
    cmap = {c.cid: c.name for c in storage.load_concepts()}
    epi = {row["cid"]: row for row in epidemic.dashboard().get("r0_list", [])}
    concepts = storage.load_concepts()
    dm = storage.load_mastery("s001")
    items = []
    for cid, cls_p in weak:
        if len(items) >= max(1, top_k):
            break
        name = cmap.get(cid, cid)
        out = hybrid_search("s001", name, top_k=3)
        hits = out.get("hits", [])
        e = epi.get(cid, {})
        items.append({"cid": cid, "name": name,
                      "class_mastery": round(cls_p, 2),
                      "weak_n": sum(1 for m in msts.values()
                                    if cid in m and m[cid].p < 0.4),
                      "r0": round(e.get("r0", 0), 2),
                      "infected_n": e.get("infected_n"),
                      "priority": round((1 - cls_p) * (1 + e.get("r0", 0)), 2),
                      "channels": out.get("channels", 2),
                      "embed_model": out.get("embed_model"), "hits": hits,
                      "graph_context": expand_with_graph(hits, concepts, dm)})
    items.sort(key=lambda x: -x["priority"])
    return {"tid": tid, "n_students": len(pool), "items": items,
            "rule": "教学优先级 = 班级掌握缺口 × (1 + 全校误区传播 R0)"}


@app.get("/api/student/epidemic_check")
def student_epidemic_check(sid: str, top_k: int = 4):
    """🧬 学生端误区疫情自检：我的薄弱考点（掌握<0.6）× 全校传播动力学(R0/感染数)
    × 三通道检索证据（密集哈希+CSR稀疏BM25+语义嵌入→RRF）+ GraphRAG 先修链。零 LLM。"""
    from scholarpulse import storage
    from scholarpulse.society import epidemic
    from scholarpulse.core.siliconflow_embed import hybrid_search
    from scholarpulse.core.rag_retriever import expand_with_graph
    m = storage.load_mastery(sid)
    if not m:
        return {"items": [], "msg": "暂无掌握度数据：先完成一轮学习循环"}
    cmap = {c.cid: c.name for c in storage.load_concepts()}
    epi = {row["cid"]: row for row in epidemic.dashboard().get("r0_list", [])}
    weak = sorted(((cid, rec.p) for cid, rec in m.items() if rec.p < 0.6),
                  key=lambda x: x[1])[:max(1, top_k) * 2]
    concepts = storage.load_concepts()
    items = []
    for cid, p in weak:
        if len(items) >= max(1, top_k):
            break
        name = cmap.get(cid, cid)
        out = hybrid_search("s001", name, top_k=2)
        hits = out.get("hits", [])
        e = epi.get(cid, {})
        items.append({"cid": cid, "name": name, "mastery": round(p, 2),
                      "r0": round(e.get("r0", 0), 2),
                      "infected_n": e.get("infected_n"),
                      "risk": round((1 - p) * (1 + e.get("r0", 0)), 2),
                      "channels": out.get("channels", 2),
                      "embed_model": out.get("embed_model"), "hits": hits,
                      "graph_context": expand_with_graph(hits, concepts, m)})
    items.sort(key=lambda x: -x["risk"])
    return {"sid": sid, "items": items,
            "rule": "个人风险 = 掌握缺口 × (1 + 全校传播 R0)——你的薄弱点若同时是全校高传播误区，优先补"}


@app.post("/api/quiz/ai_trap")
def quiz_ai_trap(body: dict):
    """🎭 对抗式陷阱题生成器（生成式 AI 扣题核心）：确定性题库(2880题)的对照组。
    取学生最薄弱考点 → 三通道检索课件证据 → LLM 针对『该生的错误模式』生成
    陷阱题（JSON Schema 受约束输出 + 证据必须来自检索片段）→ 服务端校验
    （选项完整/答案在选项内/引用证据），不合格题自动剔除。零落盘、无状态。"""
    from scholarpulse import storage, mastery as M
    from scholarpulse.core.siliconflow_embed import hybrid_search
    from scholarpulse.llm import get_llm
    sid = str(body.get("sid", "s001"))
    n = max(1, min(int(body.get("n", 3)), 5))
    m = storage.load_mastery(sid)
    if not m:
        return {"questions": [], "msg": "暂无掌握度数据"}
    weakest = min(m.items(), key=lambda kv: kv[1].p)[0]
    cmap = {c.cid: c for c in storage.load_concepts()}
    c = cmap.get(weakest)
    name = c.name if c else weakest
    out = hybrid_search("s001", name, top_k=3)
    evidence = "\n".join(f"- {h['text'][:100]}" for h in out.get("hits", [])) or "-（无检索证据）"
    students = storage.load_students()
    stu_name = students[sid].name if sid in students else sid
    sys = ("你是对抗式出题引擎。基于给定课件证据，为一名学生生成 n 道针对其薄弱考点的"
           "『陷阱题』：错误选项必须对应真实认知误区（概念混淆/边界条件/以偏概全），"
           "题干和解析只能使用证据中的事实。只输出JSON："
           '{"questions":[{"question":"题干","options":["A...","B...","C...","D..."],'
           '"answer":"A","trap":"陷阱对应的误区名","explanation":"≤60字解析","evidence":"引用的课件原句"}]}')
    user = (f"学生：{stu_name}；薄弱考点：{name}（{c.summary if c else ''}，掌握度"
            f"{M.avg_mastery({weakest: m[weakest]}):.2f}）\n课件证据：\n{evidence}\n生成 {n} 道陷阱题。")
    questions, note = [], ""
    try:
        res = get_llm().chat_json(sys, user, temperature=0.7, max_tokens=1400)
        for q in res.get("questions", []):
            opts = q.get("options") or []
            if (len(opts) == 4 and q.get("answer") in "ABCD" and q.get("question")
                    and q.get("explanation")):
                questions.append({"concept": name, **{k: q[k] for k in
                    ("question", "options", "answer", "trap", "explanation", "evidence")}})
    except Exception as e:
        note = f"生成失败（{e}）"
    if len(questions) < n and not note:
        note = f"生成 {len(questions)} 题通过校验（其余未过 Schema 校验被剔除）"
    return {"sid": sid, "concept": name, "channels": out.get("channels", 2),
            "questions": questions[:n], "note": note}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/eval")
def eval_page():
    """检索评测页：RAG 质量与延迟可视化。"""
    return FileResponse(os.path.join(STATIC, "eval.html"))


@app.post("/api/scheduler/run-now")
def scheduler_run_now():
    """手动触发一次15天报表生成并推送给老师。"""
    from scholarpulse import scheduler
    reports = scheduler.tick()
    return {"generated": len(reports), "reports": reports}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8321)  # 8000/8001 被学生管理系统占用，本项目固定 8321
