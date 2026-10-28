"""
掌握度热力图 / 进步曲线 HTML。
创新点37：把抽象的掌握度画成颜色块，学生和老师一眼看到哪里红哪里绿。
"""
from __future__ import annotations
import os
from ..models import Concept, Mastery

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def _color(p: float) -> str:
    if p < 0.4: return "#e74c3c"
    if p < 0.6: return "#e67e22"
    if p < 0.8: return "#f1c40f"
    return "#2ecc71"


def render_heatmap(sid: str, concepts: list[Concept], mastery: dict[str, Mastery]) -> str:
    cells = ""
    for cid, m in mastery.items():
        c = next((c for c in concepts if c.cid == cid), None)
        name = c.name if c else cid
        cells += (f'<div style="display:inline-block;margin:6px;padding:12px;border-radius:8px;'
                  f'background:{_color(m.p)};color:#fff;width:150px;text-align:center;">'
                  f'<b>{name}</b><br>{m.p*100:.0f}%</div>')
    html = f"""<!doctype html><meta charset="utf-8"><title>掌握热力图 {sid}</title>
<body style="font-family:sans-serif;max-width:760px;margin:30px auto;">
<h2>{sid} 考点掌握热力图</h2><p>红=薄弱 橙=待巩固 黄=良好 绿=已掌握</p>{cells}</body>"""
    path = os.path.join(OUT_DIR, f"heatmap_{sid}.html")
    open(path, "w", encoding="utf-8").write(html)
    return path
