"""多课程管理：一个学生可以选多门课，每门课独立图谱与掌握度。"""
from __future__ import annotations
import os
from .. import storage

COURSES_FILE = os.path.join(storage.DATA_DIR, "courses.json")


def list_courses() -> dict:
    import json
    if not os.path.exists(COURSES_FILE):
        return {}
    return json.load(open(COURSES_FILE, encoding="utf-8"))


def add_course(code: str, name: str, exam_date: str) -> None:
    import json
    data = list_courses()
    data[code] = {"name": name, "exam_date": exam_date}
    json.dump(data, open(COURSES_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def days_to_exam(code: str) -> int | None:
    from datetime import date
    c = list_courses().get(code)
    if not c:
        return None
    exam = date.fromisoformat(c["exam_date"])
    return (exam - date.today()).days
