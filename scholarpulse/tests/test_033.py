"""智能体三属性增强测试（全离线，不调真实 LLM）：
- 自主性：bind_tools JSON 工具协议 + create_react_agent 多步闭环 + 副官降级
- 自省性：治理 Reflexion 评审官两轮回环 + llm_calls 口径
- 社会性：政策听证委员会三派两轮辩论 + 规则降级裁决
"""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

import scholarpulse.lc_model as lcm
import scholarpulse.llm as llm_mod
from scholarpulse.lc_model import RouterChatModel
from scholarpulse.models import PolicyExperiment
from scholarpulse.graph import hierarchy, policy_debate
import scholarpulse.graph.chief as chief


class ScriptChat:
    """按脚本顺序返回 chat 文本，记录 system 供断言。"""
    def __init__(self, seq):
        self.seq = list(seq)
        self.systems = []

    def chat(self, system, user, **k):
        self.systems.append(system)
        return self.seq.pop(0)


class ScriptJson:
    """按脚本顺序返回 chat_json，记录每次 user。"""
    def __init__(self, seq):
        self.seq = list(seq)
        self.users = []

    def chat_json(self, system, user, **k):
        self.users.append(user)
        return self.seq.pop(0)

    def chat(self, *a, **k):
        raise RuntimeError("LLM off")


@tool
def add_one(n: int) -> int:
    """把 n 加一。"""
    return n + 1


# ---------- 自主性 1：JSON 工具协议解析与合成 ----------
def test_bind_tools_and_protocol(monkeypatch):
    fake = ScriptChat(['{"tool":"add_one","args":{"n":1}}',
                       '{"answer":"两次结果是 3"}',
                       '我不会，这是纯文本'])
    monkeypatch.setattr(lcm, "get_llm", lambda: fake)
    base = RouterChatModel()
    m = base.bind_tools([add_one])
    assert m is not base and len(m.bound_tools) == 1
    assert m.bound_tools[0]["function"]["name"] == "add_one"

    # ① 模型选工具 → 合成标准 tool_calls
    g1 = m._generate([SystemMessage("s"), HumanMessage("算一下")])
    tc = g1.generations[0].message.tool_calls
    assert tc and tc[0]["name"] == "add_one" and tc[0]["args"] == {"n": 1} and tc[0]["id"]
    assert "add_one" in fake.systems[0], "工具菜单必须注入 system"

    # ② 带工具历史 → answer 收口
    hist = [HumanMessage("算一下"),
            AIMessage(content="", tool_calls=[{"name": "add_one", "args": {"n": 1}, "id": "c1"}]),
            ToolMessage(content="2", name="add_one", tool_call_id="c1")]
    g2 = m._generate(hist)
    assert g2.generations[0].message.content == "两次结果是 3"
    # 扁平化必须包含工具返回行
    assert "[工具 add_one 返回] 2" in RouterChatModel._flatten(hist)

    # ③ 解析失败 → 纯文本降级（自然终止 ReAct 循环）
    g3 = m._generate([HumanMessage("q")])
    assert g3.generations[0].message.content == "我不会，这是纯文本"


def test_parse_tool_json_variants():
    p = RouterChatModel._parse_tool_json
    assert p('{"tool":"run_abm","args":{"days":5}}') == ("run_abm", {"days": 5})
    assert p('```json\n{"answer":"ok"}\n```') == ("__ANSWER__", "ok")
    assert p('前缀噪声 {"tool":"a"} 后缀') == ("a", {})
    assert p('{"tool":"a","args":"not json"}') == ("a", {})
    assert p('{"answer":"只答"}') == ("__ANSWER__", "只答")
    assert p("不是json") is None
    # 截断的 JSON（max_tokens 截尾）→ 抠出 answer 前缀而非整段外壳
    cut = '{"answer":"第一行内容\n第二行被截断'
    assert RouterChatModel._parse_tool_json(cut) == ("__ANSWER__", "第一行内容\n第二行被截断")
    # tool JSON 截断 → 抠出工具名 + 空 args（由工具报错触发模型观察重试）
    assert RouterChatModel._parse_tool_json('{"tool":"policy_sandbox","args":{"policy_te') \
        == ("policy_sandbox", {})
    assert RouterChatModel._parse_tool_json('{"tool":"a","args":') == ("a", {})
    assert RouterChatModel._parse_tool_json('{"ans') is None


# ---------- 自主性 2：create_react_agent 多步闭环（离线脚本模型） ----------
def test_react_loop_offline(monkeypatch):
    fake = ScriptChat(['{"tool":"add_one","args":{"n":1}}',
                       '{"tool":"add_one","args":{"n":2}}',
                       '{"answer":"两次加一的结果是 3"}'])
    monkeypatch.setattr(lcm, "get_llm", lambda: fake)
    from langgraph.prebuilt import create_react_agent
    model = RouterChatModel().bind_tools([add_one])
    try:
        agent = create_react_agent(model, [add_one], prompt="你是计算器。")
    except TypeError:
        agent = create_react_agent(model, [add_one], state_modifier="你是计算器。")
    final = agent.invoke({"messages": [("user", "从 0 开始加两次一")]})
    tool_msgs = [x for x in final["messages"] if x.__class__.__name__ == "ToolMessage"]
    assert len(tool_msgs) == 2, "ReAct 必须两步调用工具"
    final_ai = [x for x in final["messages"]
                if x.__class__.__name__ == "AIMessage" and x.content][-1]
    assert "3" in final_ai.content


def test_chief_fallback(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no agent")
    monkeypatch.setattr(chief, "_build_agent", _boom)
    monkeypatch.setattr(llm_mod, "get_llm", _boom)   # 副官降级链内部动态 import llm.get_llm
    r = chief.ask_chief("你好")
    assert r["agent"] == "ChiefOfStaff-Fallback"
    assert "暂时离线" in r["answer"] and r["steps"] == []


# ---------- 教师端强化 1：班级级诊断工具（离线构造数据） ----------
def test_class_tools_offline(tmp_path, monkeypatch):
    import json
    from scholarpulse import storage
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    (tmp_path / "concepts.json").write_text(json.dumps([
        {"cid": "c1", "name": "死锁"}, {"cid": "c2", "name": "进程"}]), encoding="utf-8")
    (tmp_path / "students.json").write_text(json.dumps({
        "s001": {"sid": "s001", "name": "甲", "teacher_id": "T001"},
        "s002": {"sid": "s002", "name": "乙", "teacher_id": "T001"},
        "s003": {"sid": "s003", "name": "丙", "teacher_id": "T002"}}), encoding="utf-8")
    (tmp_path / "mastery_s001.json").write_text(json.dumps({
        "c1": {"cid": "c1", "p": 0.1}, "c2": {"cid": "c2", "p": 0.7}}), encoding="utf-8")
    (tmp_path / "mastery_s002.json").write_text(json.dumps({
        "c1": {"cid": "c1", "p": 0.4}, "c2": {"cid": "c2", "p": 0.9}}), encoding="utf-8")
    (tmp_path / "mastery_s003.json").write_text(json.dumps({
        "c1": {"cid": "c1", "p": 0.9}, "c2": {"cid": "c2", "p": 0.9}}), encoding="utf-8")

    from scholarpulse.tools_registry import class_weak_map, class_risk_scan
    wm = class_weak_map.invoke({"tid": "T001"})
    assert wm["scope"] == "T001" and wm["n_students"] == 2
    assert wm["weak_top5"][0]["concept"] == "死锁"          # 班均 0.25 全班最低
    assert abs(wm["weak_top5"][0]["avg"] - 0.25) < 1e-9

    rs = class_risk_scan.invoke({"tid": "T001"})
    assert rs["scanned"] == 2 and rs["n_at_risk"] == 1       # 仅 s001（均分 0.4 < 0.5）
    assert [r["sid"] for r in rs["top10"]] == ["s001", "s002"]
    assert rs["top10"][0]["weakest"] == ["死锁"]

    rs_all = class_risk_scan.invoke({})                       # 空 tid = 全校
    assert rs_all["scanned"] == 3 and rs_all["n_at_risk"] == 1
    wm_fb = class_weak_map.invoke({"tid": "T999"})            # 无此班 → 全校口径回退
    assert wm_fb["scope"] == "全校" and wm_fb["n_students"] == 3


# ---------- 教师端强化 2：多轮会话记忆（历史重放无 tool_calls） ----------
def test_ask_chief_history_replay(monkeypatch):
    captured = {}

    class FakeAgent:
        def invoke(self, state):
            captured["msgs"] = state["messages"]
            return {"messages": [AIMessage(content="已结合上文回答。")]}

    monkeypatch.setattr(chief, "_build_agent", lambda *a, **k: FakeAgent())
    r = chief.ask_chief("给 s001 出道题", role="teacher",
                        history=[("s001 最薄弱是什么？", "死锁（0.30）。")])
    msgs = captured["msgs"]
    kinds = [type(m).__name__ for m in msgs]
    assert kinds == ["HumanMessage", "AIMessage", "HumanMessage"], "历史重放为纯文本问答对"
    assert all(not getattr(m, "tool_calls", None) for m in msgs), "重放消息不得携带 tool_calls"
    assert msgs[-1].content == "给 s001 出道题"
    assert r["answer"] == "已结合上文回答。"


# ---------- 教师端社会性创新：教研会诊（LLM 全故障 → 规则降级仍完整） ----------
def test_consult_degrade(tmp_path, monkeypatch):
    import json
    from scholarpulse import storage
    from scholarpulse.graph import consult

    def boom(*a, **k):
        raise RuntimeError("LLM off")
    monkeypatch.setattr(consult, "get_llm", boom)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    (tmp_path / "concepts.json").write_text(json.dumps([
        {"cid": "c1", "name": "死锁"}]), encoding="utf-8")
    (tmp_path / "students.json").write_text(json.dumps({
        "s001": {"sid": "s001", "name": "甲", "teacher_id": "T001"},
        "s002": {"sid": "s002", "name": "乙", "teacher_id": "T001"}}), encoding="utf-8")
    (tmp_path / "mastery_s001.json").write_text(json.dumps(
        {"c1": {"cid": "c1", "p": 0.2}}), encoding="utf-8")
    (tmp_path / "mastery_s002.json").write_text(json.dumps(
        {"c1": {"cid": "c1", "p": 0.8}}), encoding="utf-8")

    r = consult.run_consult("s001")
    assert [v["role"] for v in r["views"]] == ["本班导师", "年级教研组长", "学习科学专家"]
    assert all("规则视图" in v["view"] for v in r["views"]), "LLM 全故障必须逐角色规则降级"
    assert "规则共识" in r["consensus"] and "死锁" in r["consensus"]
    assert r["brief"] and "2/2" in r["brief"], "班内排名 2/2（均分 0.2 低于同伴 0.8，降序排名）"
    assert r["n_advisors"] == 4

    with pytest.raises(ValueError):
        consult.run_consult("no_such_sid")


# ---------- 自省性：治理 Reflexion 回环 ----------
def test_governance_reflexion_two_rounds(monkeypatch):
    fake = ScriptJson([
        {"analysis": "A1", "action": "B1"},              # deep_analyze（top_k=1）
        {"global_report": "REPORT_V1"},                  # 总监初稿
        {"ok": False, "critique": "请补充具体数字"},       # 评审官驳回 → 回环
        {"global_report": "REPORT_V2"},                  # 总监修订稿
        {"ok": True, "critique": ""},                    # 评审官通过
    ])
    monkeypatch.setattr(hierarchy, "get_llm", lambda: fake)
    rep = hierarchy.run_governance(top_k=1)
    assert rep["rounds"] == 2 and rep["global_report"] == "REPORT_V2"
    assert rep["critique"] == "" and rep["llm_calls"] == 3   # 1 深析 + 2 轮报告
    assert "修订意见" in fake.users[3], "第二轮总监必须携带评审意见重写"


def test_governance_reflexion_pass_first_round(monkeypatch):
    fake = ScriptJson([
        {"analysis": "A", "action": "B"},
        {"global_report": "REPORT_OK"},
        {"ok": True, "critique": ""},
    ])
    monkeypatch.setattr(hierarchy, "get_llm", lambda: fake)
    rep = hierarchy.run_governance(top_k=1)
    assert rep["rounds"] == 1 and rep["llm_calls"] == 2 and rep["critique"] == ""


# ---------- 社会性：政策听证委员会 ----------
def test_debate_degrade_and_guard(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LLM off")
    monkeypatch.setattr(policy_debate, "get_llm", boom)
    exp = PolicyExperiment(
        policy_text="测试政策",
        effect={"delta_avg": 0.05, "ci95": [0.04, 0.06],
                "by_tier": {"low": 0.07}, "spillover_avg": 0.0})
    r = policy_debate.run_debate(exp)
    assert [s["round"] for s in r["speeches"]] == [1, 1, 1, 2, 2, 2], "三派两轮顺序固定"
    assert {s["faction"] for s in r["speeches"]} == {"efficiency", "equity", "risk"}
    assert all("规则降级" in s["speech"] for s in r["speeches"])
    assert r["verdict"] == "通过" and r["conditions"], "CI 下界 >0 必须裁决通过"

    exp2 = exp.model_copy(update={"effect": dict(exp.effect, ci95=[-0.01, 0.05])})
    assert policy_debate.run_debate(exp2)["verdict"] == "有条件通过", "CI 跨 0 → 有条件通过"

    with pytest.raises(ValueError):
        policy_debate.run_debate(PolicyExperiment(policy_text="x", effect={}))


































