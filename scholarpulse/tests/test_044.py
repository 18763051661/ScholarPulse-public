"""班级教研官智能体类型安全测试（全离线）：
- 回归：diagnose_at_risk_students 曾把 cmap 建成 {cid: 名字字符串} 又访问 .name，
  触发 'str' object has no attribute 'name'，异常冲垮整个 ReAct 运行（steps 清空，
  前端表现为"未进行任何工具调用即失败"）。
- 边界：对象/字符串/dict/None/缺属性 五类考点数据形态 + 未知 cid 回退。
- 全流程：离线脚本模型驱动 ReAct 完整跑通 班级教研官 三工具取证 → 画像输出。
"""
from types import SimpleNamespace as NS

import scholarpulse.lc_model as lcm
from scholarpulse import storage
from scholarpulse.agents.teacher_intelligence_agent import (
    _concept_name, diagnose_at_risk_students, run_class_teacher,
    scan_bottleneck_concepts, scan_concept_failure_clusters)

# 概念库：c1 有完整属性；c2 缺 chapter；c3 缺 name（历史脏数据模拟）
CONCEPTS = [NS(cid="c1", name="死锁", chapter="第一章"),
            NS(cid="c2", name="进程调度"),
            NS(cid="c3")]

STUDENTS = {"s001": NS(sid="s001", name="甲", teacher_id="T001"),
            "s002": NS(sid="s002", name="乙", teacher_id="T001"),
            "s003": NS(sid="s003", name="丙", teacher_id="T002")}

MASTERY = {"s001": {"c1": NS(p=0.2), "c2": NS(p=0.8), "cx": NS(p=0.45)},
           "s002": {"c1": NS(p=0.9), "c2": NS(p=0.9)},
           "s003": {"c1": NS(p=0.95), "c2": NS(p=0.85)}}

ANSWERS = [NS(cid="c1", ts=1, correct=False), NS(cid="c1", ts=2, correct=False),
           NS(cid="cx", ts=3, correct=False), NS(cid="cx", ts=4, correct=False)]


def _patch_storage(monkeypatch):
    monkeypatch.setattr(storage, "load_concepts", lambda: CONCEPTS)
    monkeypatch.setattr(storage, "load_students", lambda: STUDENTS)
    monkeypatch.setattr(storage, "load_mastery",
                        lambda sid: MASTERY.get(sid, {}))
    monkeypatch.setattr(storage, "load_answers", lambda sid: ANSWERS)


# ---------- 1. 考点名安全解析：五类数据形态 ----------
def test_concept_name_helper_shapes():
    assert _concept_name(NS(name="死锁"), "c1") == "死锁"        # 对象
    assert _concept_name("进程", "c2") == "进程"                 # 历史字符串形态
    assert _concept_name("", "c2") == "c2"                       # 空字符串回退
    assert _concept_name(None, "c3") == "c3"                     # None 回退
    assert _concept_name({"name": "页置换"}, "c4") == "页置换"   # dict 形态
    assert _concept_name({}, "c4") == "c4"                       # dict 缺键回退
    assert _concept_name(NS(), "c3") == "c3"                     # 对象缺 name 属性回退


# ---------- 2. 核心回归：卡壳考点不再触发 str.name 崩溃 ----------
def test_diagnose_stuck_concept_regression(monkeypatch):
    """s001 在 c1/cx 均有≥2次错答且掌握度<0.4 → 必须正常给出卡壳考点名。
    修复前此路径 100% 抛 AttributeError('str' object has no attribute 'name')。"""
    _patch_storage(monkeypatch)
    r = diagnose_at_risk_students.invoke({"tid": "T001"})
    assert r["at_risk_count"] == 2                      # bottom5 = s001(0.43) + s002(0.90)
    stu = r["students"][0]                              # 按均分升序，s001 最薄弱排第一
    assert stu["name"] == "甲" and stu["sid"] == "s001"
    stuck = stu["stuck_concept"]
    assert stuck is not None
    assert stuck["name"] == "死锁"                    # 对象形态正确解析出考点名
    assert stu["root_cause"] == "卡壳于『死锁』" or "死锁" in stu["root_cause"]


def test_diagnose_unknown_cid_fallback(monkeypatch):
    """掌握度里存在概念库没有的 cid（cx）→ 名字回退为 cid，不得崩溃。"""
    _patch_storage(monkeypatch)
    # 覆盖 s001 掌握度：让概念库外的 cx 成为掌握度最低的卡壳候选
    m2 = dict(MASTERY)
    m2["s001"] = {"c1": NS(p=0.3), "c2": NS(p=0.8), "cx": NS(p=0.2)}
    monkeypatch.setattr(storage, "load_mastery", lambda sid: m2.get(sid, {}))
    r = diagnose_at_risk_students.invoke({"tid": "T001"})
    assert r["students"][0]["stuck_concept"]["name"] == "cx"


# ---------- 3. 失败聚类：混合形态 + 缺属性对象 ----------
def test_scan_clusters_mixed_shapes(monkeypatch):
    _patch_storage(monkeypatch)
    r = scan_concept_failure_clusters.invoke({"tid": "T001"})
    by_cid = {row["cid"]: row for row in r["top_failing_concepts"]}
    assert by_cid["c1"]["name"] == "死锁" and by_cid["c1"]["chapter"] == "第一章"
    assert by_cid["cx"]["name"] == "cx"               # 未知 cid 回退
    assert by_cid["c2"]["name"] == "进程调度"          # 缺 chapter 的对象不崩


# ---------- 4. 年级瓶颈：两班聚合 + 名称解析安全 ----------
def test_scan_bottleneck_name_safety(monkeypatch):
    _patch_storage(monkeypatch)
    r = scan_bottleneck_concepts.invoke({"tid": "T001"})
    names = [b["name"] for b in r["common_bottlenecks"] + r["class_specific"]]
    assert "死锁" in names and "c3" in names          # 缺 name 属性的 c3 回退为 cid
    assert all(b["chapter"] == "" or isinstance(b["chapter"], str)
               for b in r["common_bottlenecks"] + r["class_specific"])


# ---------- 5. 全流程：离线脚本模型驱动 ReAct 完整跑通班级教研官 ----------
def test_run_class_teacher_full_react_offline(monkeypatch):
    """三步脚本：调分布工具 → 调(曾崩溃的)根因诊断工具 → 收口画像结论。
    验证 ReAct 循环不再被属性错误中断，steps 完整保留。"""

    class ScriptChat:
        def __init__(self, seq): self.seq = list(seq)

        def chat(self, system, user, **k): return self.seq.pop(0)

    _patch_storage(monkeypatch)
    fake = ScriptChat(['{"tool":"scan_class_distribution","args":{"tid":"T001"}}',
                       '{"tool":"diagnose_at_risk_students","args":{"tid":"T001"}}',
                       '{"answer":"班级画像：分布对称但1名学生卡壳于死锁与cx，需分层介入"}'])
    monkeypatch.setattr(lcm, "get_llm", lambda: fake)
    r = run_class_teacher("T001")
    assert "失败" not in r["answer"], r["answer"]
    assert len(r["steps"]) == 2, r["steps"]
    assert r["steps"][0]["tool"] == "scan_class_distribution"
    assert r["steps"][1]["tool"] == "diagnose_at_risk_students"
    assert all(s.get("result") for s in r["steps"])   # 每步都有工具观察
    assert "班级画像" in r["answer"] and "死锁" in r["answer"]
