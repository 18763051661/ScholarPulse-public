"""学习报告导出器测试（全离线）：思维导图/饼图/四格式导出/笔记存取。"""
from types import SimpleNamespace as NS

from scholarpulse.core import report_exporter as RE
from scholarpulse.core.report_exporter import (
    build_mindmap, build_pie, export_md, export_html, export_pdf, export_docx,
)
from scholarpulse import storage

CONCEPTS = [NS(cid="c1", name="子网掩码", chapter="第一章", course="net"),
            NS(cid="c2", name="TCP握手", chapter="第一章", course="net"),
            NS(cid="c3", name="路由转发", chapter="第二章", course="net")]
MASTERY = {"c1": NS(p=0.85), "c2": NS(p=0.5), "c3": NS(p=0.2)}

REPORT = {"sid": "t999", "generated_at": "2026-10-04 10:00",
          "overview": {"n_concepts": 3, "avg": 0.52,
                       "tiers": {"基础(<0.4)": 1, "巩固(0.4-0.7)": 1, "拓展(>0.7)": 1}},
          "weak": [{"name": "TCP握手", "p": 0.5}, {"name": "路由转发", "p": 0.2}],
          "wrong_book": [{"concept": "路由转发", "n": 3, "last_q": "路由表匹配最长前缀??"}],
          "notes": [{"t": "2026-10-04 09:00", "text": "子网掩码=区分网络位与主机位"}]}


def test_mindmap_tree_structure(monkeypatch):
    monkeypatch.setattr(storage, "load_students", lambda: {})
    tree = build_mindmap(CONCEPTS, MASTERY, "t999")
    assert tree["name"].endswith("知识树")
    # 四层结构：根 → 学科 → 章节 → 概念（→ 建议叶）
    courses = {co["name"]: co for co in tree["children"]}
    assert set(courses) == {"计算机网络"}                     # 学科层（course 展示名）
    chapters = {ch["name"]: ch for ch in courses["计算机网络"]["children"]}
    assert set(chapters) == {"第一章", "第二章"}              # 章节分组
    c1_node = next(n for n in chapters["第一章"]["children"] if "子网掩码" in n["name"])
    assert "children" not in c1_node                     # 达标概念不挂建议叶
    # 薄弱优先排序：第二章内已测概念按掌握度升序，未测殿后
    ch2 = chapters["第二章"]["children"]
    assert ch2[0]["mastery"] == 0.2 and "重修基础" in ch2[0]["children"][0]["name"]
    # 未测概念仍入图（覆盖完整学科面），mastery 为 None
    untested = build_mindmap(CONCEPTS, {}, "t999")["children"][0]["children"][0]["children"]
    assert all(n["mastery"] is None and "children" not in n for n in untested)


def test_pie_normalizes_to_100(monkeypatch):
    monkeypatch.setattr(storage, "load_students", lambda: {})
    pie = build_pie(MASTERY)
    assert {d["name"] for d in pie} == {"基础(<0.4)", "巩固(0.4-0.7)", "拓展(>0.7)"}
    assert abs(sum(d["pct"] for d in pie) - 100.0) < 0.3  # 比例归一
    assert {d["n"] for d in pie} == {1}


def test_export_md_and_html_content():
    md = export_md(REPORT)
    assert md.startswith("# ScholarPulse 学习报告")
    assert "一、掌握度概览" in md and "路由转发×3" in md
    assert "子网掩码=区分网络位与主机位" in md            # 笔记进入报告
    html = export_html(REPORT)
    assert html.startswith("<!doctype html>") and "<h2>三、错题本" in html


def test_export_pdf_file(tmp_path):
    path = export_pdf(REPORT, out_dir=str(tmp_path))
    import os
    assert os.path.exists(path) and os.path.getsize(path) > 1000


def test_export_docx_file(tmp_path):
    path = export_docx(REPORT, out_dir=str(tmp_path))
    import os
    assert os.path.exists(path) and os.path.getsize(path) > 1000
    import docx
    d = docx.Document(path)
    assert any("错题本" in p.text for p in d.paragraphs)  # DOCX 内容可回读


def test_export_report_dispatcher(monkeypatch, tmp_path):
    monkeypatch.setattr(RE, "EXPORT_DIR", str(tmp_path))
    r = RE.export_report("t999", "md")
    assert r["fmt"] == "md" and r["size"] > 0
    bad = RE.export_report("t999", "xlsx")
    assert "error" in bad                                 # 非法格式被拒


def test_notes_roundtrip_and_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    for i in range(55):
        storage.save_notes("t888", storage.load_notes("t888") + [{"t": "t", "text": f"n{i}"}])
    notes = storage.load_notes("t888")
    assert len(notes) == 50                               # 上限 50 条
    assert notes[-1]["text"] == "n54"                     # 保留最新的
