"""学习报告多格式导出（MD / HTML / PDF / DOCX）+ 思维导图/饼图数据层。

对标需求：
  - 学习管理可视化：进度/错题/笔记/仪表板，支持 PDF/DOCX/MD/HTML 多格式导出
  - 自动绘制可流动的思维导图与饼图，层层推进把控学生

设计：
  - 数据聚合 aggregate(sid) —— 掌握度概览/薄弱点/错题本/学生笔记，全部来自
    storage 现有数据，不重复存档
  - build_mindmap —— 章节→概念→补弱建议 三层思维导图树（"层层推进"），
    概念节点带掌握度，薄弱概念自动挂补弱建议叶子
  - build_pie —— 基础/巩固/拓展 三层分布饼图数据（比例归一）
  - 四格式导出 —— reportlab（PDF）与 python-docx（DOCX）项目已有依赖，
    MD/HTML 纯文本生成，零新增依赖；全部纯函数可离线测试
"""
from __future__ import annotations
import os
from datetime import datetime

from .. import storage
from ..storage import DATA_DIR

EXPORT_DIR = os.path.join(DATA_DIR, "exports")


# ============================================================
# 数据聚合
# ============================================================

def _tiers(mastery: dict) -> dict:
    t = {"基础(<0.4)": 0, "巩固(0.4-0.7)": 0, "拓展(>0.7)": 0}
    for mm in mastery.values():
        if mm.p < 0.4:
            t["基础(<0.4)"] += 1
        elif mm.p <= 0.7:
            t["巩固(0.4-0.7)"] += 1
        else:
            t["拓展(>0.7)"] += 1
    return t


def build_pie(mastery: dict) -> list[dict]:
    """掌握度三分层饼图数据（比例归一到 100）。"""
    tiers = _tiers(mastery)
    n = sum(tiers.values()) or 1
    return [{"name": k, "n": v, "pct": round(v / n * 100, 1)}
            for k, v in tiers.items()]


def _advice(p: float) -> str:
    if p < 0.4:
        return "建议：回退上游先修概念重修基础"
    if p < 0.6:
        return "建议：针对性练习巩固 + 艾宾浩斯复习"
    return "建议：综合迁移题拔高"


# 课程代号 → 展示名：以 curriculum.COURSES 为准（自动覆盖全部已装课程），
# 保留早期 legacy 代号别名，防止旧数据回退显示代号
from .curriculum import COURSES as _CURRICULUM_COURSES
COURSE_NAMES = {**_CURRICULUM_COURSES, "coa": "计算机组成原理", "math": "离散数学",
                "prog": "程序设计基础", "ai": "人工智能导论"}


def build_mindmap(concepts: list, mastery: dict, sid: str) -> dict:
    """学科→章节→概念→补弱建议 四层思维导图，覆盖计算机专业全部学科。

    - 概念节点带掌握度（未测为 None，仍入图——保证图谱覆盖完整学科面）
    - 已测且 <0.6 的薄弱概念自动挂补弱建议叶子（"层层推进"）
    - 同章节内排序：薄弱优先（升序），未测概念殿后
    """
    students = storage.load_students()
    sname = students.get(sid).name if sid in students else sid
    tree: dict[str, dict[str, list]] = {}
    for c in concepts:
        mm = mastery.get(c.cid)
        p = mm.p if mm is not None else None
        node = {"cid": c.cid,
                "name": c.name if p is None else f"{c.name}（掌握{round(p * 100)}%）",
                "mastery": round(p, 2) if p is not None else None}
        if p is not None and p < 0.6:
            node["children"] = [{"name": _advice(p)}]
        course = COURSE_NAMES.get(getattr(c, "course", "os"), getattr(c, "course", "os"))
        chap = (getattr(c, "chapter", "") or "未分组")
        tree.setdefault(course, {}).setdefault(chap, []).append(node)
    children = []
    for course in sorted(tree):
        chaps = []
        for chname, nodes in sorted(tree[course].items()):
            nodes.sort(key=lambda x: (x["mastery"] is None,
                                      x["mastery"] if x["mastery"] is not None else 1.0))
            chaps.append({"name": chname, "children": nodes})
        children.append({"name": course, "children": chaps})
    return {"name": f"{sname} · 知识树", "children": children}


def aggregate(sid: str) -> dict:
    """学习报告数据聚合：概览/薄弱点/错题本/笔记/思维导图/饼图。"""
    concepts = storage.load_concepts()
    mastery = storage.load_mastery(sid)
    cmap_name = {c.cid: c.name for c in concepts}
    # 概览
    avg = round(sum(mm.p for mm in mastery.values()) / len(mastery), 2) if mastery else 0.0
    # 薄弱点（升序取前8）
    weak = sorted(((cmap_name.get(cid, cid), mm.p) for cid, mm in mastery.items()
                   if mm.p < 0.6), key=lambda x: x[1])[:8]
    # 错题本：按考点聚合错误次数 + 最近错题原文
    wrong: dict[str, dict] = {}
    for a in storage.load_answers(sid):
        if a.correct:
            continue
        row = wrong.setdefault(a.cid, {"n": 0, "last_q": ""})
        row["n"] += 1
        row["last_q"] = (a.question or "")[:50]
    wrong_rows = sorted(wrong.items(), key=lambda x: -x[1]["n"])[:8]
    return {
        "sid": sid,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "overview": {"n_concepts": len(mastery), "avg": avg,
                     "tiers": _tiers(mastery)},
        "weak": [{"name": n, "p": round(p, 2)} for n, p in weak],
        "wrong_book": [{"concept": cmap_name.get(cid, cid), "n": d["n"],
                        "last_q": d["last_q"]} for cid, d in wrong_rows],
        "notes": storage.load_notes(sid),
        "mindmap": build_mindmap(concepts, mastery, sid),
        "pie": build_pie(mastery),
    }


# ============================================================
# 四格式导出（MD / HTML / PDF / DOCX）
# ============================================================

def _sections(r: dict) -> list[tuple[str, list[str]]]:
    """报告正文（四格式共用内容源）。"""
    ov = r["overview"]
    secs = [
        ("一、掌握度概览", [
            f"覆盖考点 {ov['n_concepts']} 个，平均掌握度 {ov['avg']}。",
            "分层分布：" + "；".join(f"{k} {v} 人次" for k, v in ov["tiers"].items()),
        ]),
        ("二、薄弱点（按掌握度升序）", [f"{w['name']}：{w['p']}" for w in r["weak"]] or ["无（全部达标）"]),
        ("三、错题本（按错误次数）", [f"{w['concept']}×{w['n']}（最近错：{w['last_q']}）" for w in r["wrong_book"]] or ["错题本为空"]),
        ("四、学习笔记", [n["text"] for n in r["notes"]] or ["暂无笔记"]),
    ]
    return secs


def export_md(r: dict) -> str:
    lines = [f"# ScholarPulse 学习报告 · {r['sid']}", f"> 生成时间：{r['generated_at']}", ""]
    for title, items in _sections(r):
        lines.append(f"## {title}")
        lines += [f"- {i}" for i in items] + [""]
    return "\n".join(lines)


def export_html(r: dict) -> str:
    body = ""
    for title, items in _sections(r):
        body += f"<h2>{title}</h2><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    return ("<!doctype html><meta charset='utf-8'><title>ScholarPulse 学习报告</title>"
            "<style>body{font-family:'Microsoft YaHei',sans-serif;max-width:760px;"
            "margin:40px auto;color:#1a1a2e}h1{color:#4a1a7a}h2{color:#6b2da0;"
            "border-left:4px solid #8e44ad;padding-left:8px}</style>"
            f"<h1>ScholarPulse 学习报告 · {r['sid']}</h1>"
            f"<p>生成时间：{r['generated_at']}</p>{body}")


def export_pdf(r: dict, out_dir: str | None = None) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    try:
        pdfmetrics.registerFont(TTFont('MSYH', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
    except Exception:
        pass  # 已注册或字体缺失时回退默认字体
    out_dir = out_dir or EXPORT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"report_{r['sid']}.pdf")
    body_style = ParagraphStyle('b', fontName='MSYH', fontSize=10.5, leading=17)
    h_style = ParagraphStyle('h', fontName='MSYH', fontSize=14, leading=20,
                             spaceBefore=12, spaceAfter=6)
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = [Paragraph(f"ScholarPulse 学习报告 · {r['sid']}", h_style),
             Paragraph(f"生成时间：{r['generated_at']}", body_style), Spacer(1, 8)]
    for title, items in _sections(r):
        story.append(Paragraph(title, h_style))
        story += [Paragraph(f"• {i}", body_style) for i in items]
    doc.build(story)
    return path


def export_docx(r: dict, out_dir: str | None = None) -> str:
    import docx
    out_dir = out_dir or EXPORT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"report_{r['sid']}.docx")
    d = docx.Document()
    d.add_heading(f"ScholarPulse 学习报告 · {r['sid']}", 0)
    d.add_paragraph(f"生成时间：{r['generated_at']}")
    for title, items in _sections(r):
        d.add_heading(title, 1)
        for i in items:
            d.add_paragraph(i, style="List Bullet")
    d.save(path)
    return path


def export_report(sid: str, fmt: str) -> dict:
    """统一导出入口：fmt ∈ md/html/pdf/docx。"""
    r = aggregate(sid)
    fmt = fmt.lower()
    if fmt == "md":
        out_dir = EXPORT_DIR
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"report_{sid}.md")
        open(path, "w", encoding="utf-8").write(export_md(r))
    elif fmt == "html":
        out_dir = EXPORT_DIR
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"report_{sid}.html")
        open(path, "w", encoding="utf-8").write(export_html(r))
    elif fmt == "pdf":
        path = export_pdf(r)
    elif fmt == "docx":
        path = export_docx(r)
    else:
        return {"error": f"不支持的格式：{fmt}（可选 md/html/pdf/docx）"}
    return {"sid": sid, "fmt": fmt, "path": path, "size": os.path.getsize(path)}
