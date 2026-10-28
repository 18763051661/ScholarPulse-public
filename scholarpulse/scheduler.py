"""
定时报表调度：每 15 天自动为每个学生生成学习分析报表，
并推送到其指定指导老师（teacher_id）。
用后台线程实现，无额外依赖。
"""
from __future__ import annotations
import os, json, threading, time
from datetime import datetime, timedelta

REPORT_DIR = "data/reports"
INTERVAL_DAYS = 15
STATE_FILE = "data/scheduler_state.json"


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE, encoding="utf-8"))
    return {}


def _save_state(s: dict):
    os.makedirs("data", exist_ok=True)
    json.dump(s, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def run_report_for_student(sid: str, teacher_id: str) -> dict:
    """生成单个学生的15天报表并推送给老师。"""
    from . import storage
    from . import mastery as M
    os.makedirs(REPORT_DIR, exist_ok=True)
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    avg = M.avg_mastery(mastery)
    weak = M.weak_concepts(mastery, concepts)
    report = {
        "sid": sid,
        "teacher_id": teacher_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period_days": INTERVAL_DAYS,
        "avg_mastery": round(avg, 2),
        "weakest": [{"name": c.name, "mastery": round(mastery[c.cid].p, 2)} for c in weak[:5]],
        "action": f"建议老师关注 {sid}：最薄弱考点 {weak[0].name if weak else '无'}，建议安排针对性练习。",
    }
    fname = f"{REPORT_DIR}/{sid}_{datetime.now().strftime('%Y%m%d')}.json"
    json.dump(report, open(fname, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 推送到老师收件箱
    inbox = f"data/teacher_inbox/{teacher_id}"
    os.makedirs(inbox, exist_ok=True)
    open(f"{inbox}/{sid}_report.json", "w", encoding="utf-8").write(
        json.dumps(report, ensure_ascii=False, indent=2))
    return report


def tick():
    """检查哪些学生到期需要生成报表。"""
    from . import storage
    state = _load_state()
    students = storage.load_students()
    results = []
    for sid, s in students.items():
        last = state.get(sid)
        due = (last is None) or (datetime.fromisoformat(last) < datetime.now() - timedelta(days=INTERVAL_DAYS))
        if due:
            r = run_report_for_student(sid, s.teacher_id)
            state[sid] = datetime.now().isoformat(timespec="seconds")
            results.append(r)
    if results:
        _save_state(state)
    return results


def start_background():
    """启动后台线程，每小时检查一次是否有学生到期。"""
    def loop():
        while True:
            try:
                tick()
            except Exception as e:
                print("[scheduler]", e)
            time.sleep(3600)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
