"""把学情报告渲染成单文件 HTML，可直接在浏览器打开，也方便老师后端嵌入。"""
from __future__ import annotations
import os
from .models import Concept, Mastery, Student
from .agents import report_agent

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def bar(pct: float) -> str:
    color = "#e74c3c" if pct < 0.6 else "#f39c12" if pct < 0.8 else "#27ae60"
    return f'<div style="background:#eee;border-radius:6px;width:160px;height:10px;"><div style="width:{pct*100:.0f}%;background:{color};height:10px;border-radius:6px;"></div></div>'


def render_student_html(student: Student, concepts: list[Concept],
                        mastery: dict[str, Mastery], report: dict) -> str:
    cmap = {c.cid: c for c in concepts}
    rows = ""
    for cid, m in mastery.items():
        c = cmap.get(cid)
        name = c.name if c else cid
        weight = f"{c.exam_weight:.2f}" if c else "-"
        rows += f"<tr><td>{name}</td><td>{weight}</td><td>{m.p*100:.0f}%</td><td>{bar(m.p)}</td></tr>"
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{student.name} 的学情报告</title>
<style>body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;max-width:820px;margin:30px auto;color:#222;}}
table{{border-collapse:collapse;width:100%;margin:16px 0;}}td,th{{border:1px solid #ddd;padding:8px;}}
.badge{{display:inline-block;padding:2px 10px;border-radius:10px;background:#3498db;color:#fff;}}
.risk{{color:#e74c3c;font-weight:bold;}}</style></head><body>
<h1>📚 ScholarPulse 学情报告</h1>
<p>学生：<b>{student.name}</b> <span class="badge">{student.tier}</span>　指导老师编号：{student.teacher_id or '未分配'}</p>
<h3>总评</h3><p>{report.get('summary','')}</p>
<p>考试风险：<span class="risk">{report.get('exam_risk','-')}</span>　|　下一步重点：{report.get('next_focus','-')}</p>
<h3>考点掌握度</h3>
<table><tr><th>考点</th><th>考试权重</th><th>掌握度</th><th>进度</th></tr>{rows}</table>
<p style="color:#888;font-size:12px;">由多智能体系统自动生成 · 反幻觉校验已启用</p>
</body></html>"""
    path = os.path.join(OUT_DIR, f"report_{student.sid}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
