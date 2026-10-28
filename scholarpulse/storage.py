"""JSON 文件存储层：零依赖，方便在 IDE 里直接打开查看数据。"""
from __future__ import annotations
import json
import os
from datetime import datetime
from .models import (Concept, Mastery, Student, QuizItem, AnswerRecord, ReportSnapshot,
                     TwinProfile, TwinBattle, Teacher, Society, GovernanceReport,
                     PolicyExperiment, PolicyDebate)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def _read_json(name: str, default):
    p = _path(name)
    if not os.path.exists(p):
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(name: str, data) -> None:
    with open(_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- 知识图谱 ----------
def load_concepts() -> list[Concept]:
    raw = _read_json("concepts.json", [])
    return [Concept(**c) for c in raw]


def save_concepts(concepts: list[Concept]) -> None:
    _write_json("concepts.json", [c.model_dump() for c in concepts])


# ---------- 学生 ----------
def load_students() -> dict[str, Student]:
    raw = _read_json("students.json", {})
    return {k: Student(**v) for k, v in raw.items()}


def save_students(students: dict[str, Student]) -> None:
    _write_json("students.json", {k: v.model_dump() for k, v in students.items()})


# ---------- 掌握度 ----------
def load_mastery(sid: str) -> dict[str, Mastery]:
    raw = _read_json(f"mastery_{sid}.json", {})
    return {k: Mastery(**v) for k, v in raw.items()}


def save_mastery(sid: str, mastery: dict[str, Mastery]) -> None:
    _write_json(f"mastery_{sid}.json", {k: v.model_dump() for k, v in mastery.items()})


# ---------- 答题记录 ----------
def load_answers(sid: str) -> list[AnswerRecord]:
    raw = _read_json(f"answers_{sid}.json", [])
    return [AnswerRecord(**a) for a in raw]


def append_answers(sid: str, records: list[AnswerRecord]) -> None:
    cur = load_answers(sid)
    cur.extend(records)
    _write_json(f"answers_{sid}.json", [a.model_dump() for a in cur])


# ---------- 快照（用于 15 天纵向对比） ----------
def load_snapshots(sid: str) -> list[ReportSnapshot]:
    raw = _read_json(f"snap_{sid}.json", [])
    return [ReportSnapshot(**s) for s in raw]


def append_snapshot(sid: str, snap: ReportSnapshot) -> None:
    cur = load_snapshots(sid)
    cur.append(snap)
    _write_json(f"snap_{sid}.json", [s.model_dump() for s in cur])


# ---------- 自适应题库（CAT，AI 批量生成一次落盘） ----------
def load_quiz_bank() -> list[QuizItem]:
    raw = _read_json("quiz_bank.json", [])
    return [QuizItem(**q) for q in raw]


def save_quiz_bank(items: list[QuizItem]) -> None:
    _write_json("quiz_bank.json", [q.model_dump() for q in items])


# ---------- 测验 Elo 能力值（自适应测验的能力估计） ----------
def load_quiz_elo(sid: str) -> dict:
    return _read_json(f"quiz_elo_{sid}.json", {"elo": 1200, "n": 0})


def save_quiz_elo(sid: str, d: dict) -> None:
    _write_json(f"quiz_elo_{sid}.json", d)


# ---------- 错题检察院判决书 ----------
def load_prosecution(sid: str) -> list[dict]:
    return _read_json(f"prosecution_{sid}.json", [])


def save_prosecution(sid: str, verdicts: list[dict]) -> None:
    _write_json(f"prosecution_{sid}.json", verdicts)


# ---------- 搭子天梯赛战绩 ----------
def load_ladder(sid: str) -> dict:
    return _read_json(f"ladder_{sid}.json", {"w": 0, "l": 0, "d": 0, "history": []})


def save_ladder(sid: str, d: dict) -> None:
    _write_json(f"ladder_{sid}.json", d)


# ---------- 搭子讨论聊天室（会话记忆） ----------
def load_buddy_chat(sid: str) -> list:
    return _read_json(f"buddy_chat_{sid}.json", [])


def save_buddy_chat(sid: str, h: list) -> None:
    _write_json(f"buddy_chat_{sid}.json", h[-14:])


# ---------- 学生笔记 ----------
def load_notes(sid: str) -> list[dict]:
    """学生笔记：[{"t": 时间, "text": 内容}]，最近 50 条。"""
    return _read_json(f"notes_{sid}.json", [])


def save_notes(sid: str, notes: list[dict]) -> None:
    _write_json(f"notes_{sid}.json", notes[-50:])


# ---------- 数字孪生对抗赛 ----------
def load_twin(sid: str) -> dict:
    """孪生档案：{"profile": TwinProfile, "battles": [TwinBattle]}。"""
    raw = _read_json(f"twin_{sid}.json", {})
    profile = raw.get("profile")
    return {
        "profile": TwinProfile(**profile) if profile else None,
        "battles": [TwinBattle(**b) for b in raw.get("battles", [])],
    }


def save_twin(sid: str, profile: TwinProfile, battles: list[TwinBattle]) -> None:
    _write_json(f"twin_{sid}.json", {
        "profile": profile.model_dump(),
        "battles": [b.model_dump() for b in battles],
    })


def load_twin_pending(sid: str) -> dict | None:
    """待作答的孪生对战题（含参考答案，不下发前端）。"""
    return _read_json(f"twin_pending_{sid}.json", None)


def save_twin_pending(sid: str, pending: dict) -> None:
    _write_json(f"twin_pending_{sid}.json", pending)


def clear_twin_pending(sid: str) -> None:
    p = _path(f"twin_pending_{sid}.json")
    if os.path.exists(p):
        os.remove(p)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------- 学区级治理（L0-L4） ----------
def load_teachers() -> dict[str, Teacher]:
    raw = _read_json("teachers.json", {})
    return {k: Teacher(**v) for k, v in raw.items()}


def save_teachers(teachers: dict[str, Teacher]) -> None:
    _write_json("teachers.json", {k: v.model_dump() for k, v in teachers.items()})


def load_society() -> Society | None:
    raw = _read_json("society.json", None)
    return Society(**raw) if raw else None


def save_society(society: Society) -> None:
    _write_json("society.json", society.model_dump())


def save_answers(sid: str, records: list[AnswerRecord]) -> None:
    """整写该生答题记录（仿真批量落盘用，避免 700 次 读-改-写）。"""
    _write_json(f"answers_{sid}.json", [a.model_dump() for a in records])


def load_governance() -> GovernanceReport | None:
    raw = _read_json("governance.json", None)
    return GovernanceReport(**raw) if raw else None


def save_governance(rep: GovernanceReport) -> None:
    _write_json("governance.json", rep.model_dump())


def load_epidemic() -> dict:
    """误区疫情档案：{"states": [...], "sir_curve": [...], "history": {...}}。"""
    return _read_json("epidemic.json", {"states": [], "sir_curve": [], "history": {}})


def save_epidemic(data: dict) -> None:
    _write_json("epidemic.json", data)


def load_policy() -> PolicyExperiment | None:
    raw = _read_json("policy.json", None)
    return PolicyExperiment(**raw) if raw else None


def save_policy(exp: PolicyExperiment) -> None:
    _write_json("policy.json", exp.model_dump())


def load_debate() -> PolicyDebate | None:
    raw = _read_json("debate.json", None)
    return PolicyDebate(**raw) if raw else None


def save_debate(d: PolicyDebate) -> None:
    _write_json("debate.json", d.model_dump())
