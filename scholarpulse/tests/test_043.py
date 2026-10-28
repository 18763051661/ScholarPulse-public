"""课程库与导图回归：入库合并（防上传清库）/ 多课程显示名。"""
from types import SimpleNamespace as NS


def test_ingest_merges_not_overwrites(monkeypatch):
    """上传课件必须合并入库：旧课程保留、新课程追加、同 cid 更新。"""
    import scholarpulse.graph._pipeline as P
    from scholarpulse import storage

    monkeypatch.setattr(storage, "load_concepts", lambda: [
        NS(cid="keep1", name="旧考点", course="ds", chapter="c", summary="s",
           prerequisites=[], exam_weight=0.5, tier="core", noise="core")])
    saved = {}
    monkeypatch.setattr(storage, "save_concepts",
                        lambda cs: saved.update({c.cid: c for c in cs}))

    class FakeIA:
        @staticmethod
        def build_knowledge_graph(t, n):
            return [NS(cid="new1", name="新考点", course="os", chapter="c", summary="s",
                       prerequisites=[], exam_weight=0.5, tier="core", noise="core"),
                    NS(cid="keep1", name="旧考点改", course="ds", chapter="c", summary="s2",
                       prerequisites=[], exam_weight=0.5, tier="core", noise="core")]

    class FakeFA:
        @staticmethod
        def filter_noise(cs):
            return cs

    monkeypatch.setattr(P, "ingest_agent", FakeIA)
    monkeypatch.setattr(P, "filter_agent", FakeFA)
    st = P._node_ingest({"sid": "s1", "material_text": "x", "material_name": "n"})
    assert set(saved) == {"keep1", "new1"}        # 旧的没被清掉，新的已入库
    assert [c.cid for c in st["concepts"]] == ["new1", "keep1"]  # state 仍为本次抽取结果


def test_upload_mastery_keeps_other_courses(monkeypatch):
    """merge_ingested 只初始化新增考点，不清空学生其他课程掌握度。"""
    from types import SimpleNamespace as NSM
    import webapp.app as WA
    from scholarpulse import storage, mastery as M

    old = {"keep1": NSM(p=0.8)}
    monkeypatch.setattr(storage, "load_mastery", lambda sid: dict(old))
    captured = {}
    monkeypatch.setattr(storage, "save_mastery", lambda sid, m: captured.update(m))
    monkeypatch.setattr(storage, "load_concepts", lambda: [])
    monkeypatch.setattr(storage, "save_concepts", lambda cs: None)
    monkeypatch.setattr(M, "init_mastery",
                        lambda cs: {c.cid: NSM(p=0.0) for c in cs})
    new_c = NS(cid="fresh1", name="新考点", course="os", chapter="c", summary="s",
               prerequisites=[], exam_weight=0.5, tier="core", noise="core")
    WA.merge_ingested("s1", [new_c])
    assert "keep1" in captured and captured["keep1"].p == 0.8   # 其他课程掌握度保留
    assert "fresh1" in captured                                 # 新考点已初始化


def test_mindmap_multi_course_display_names(monkeypatch):
    """多课程导图显示中文课名而非代号。"""
    from scholarpulse.core import report_exporter as RE
    cs = [NS(cid="a1", name="A考点", course="os", chapter="第一章 x"),
          NS(cid="b1", name="B考点", course="ml", chapter="第一章 y")]
    monkeypatch.setattr(RE.storage, "load_students", lambda: {})
    r = RE.build_mindmap(cs, {}, "s9")
    names = {c["name"] for c in r["children"]}
    assert names == {"操作系统", "机器学习"}
