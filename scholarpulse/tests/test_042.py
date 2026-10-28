"""因果推断智能体离线测试：路径方向 / 确定性优先 / 根因溯源 / 空数据守卫。"""
from types import SimpleNamespace as NS

import scholarpulse.agents.causal_inference_agent as CA
from scholarpulse.agents.causal_inference_agent import (
    build_causal_graph, do_calculus, find_root_cause)

# 概念链：A ← B ← C（B 的先修是 A，C 的先修是 B）
CONCEPTS = [NS(cid="ca", name="概念A", summary="A的摘要", chapter="一", prerequisites=[]),
            NS(cid="cb", name="概念B", summary="B的摘要", chapter="一", prerequisites=["ca"]),
            NS(cid="cc", name="概念C", summary="C的摘要", chapter="二", prerequisites=["cb"])]
MASTERY = {"ca": NS(p=0.15), "cb": NS(p=0.9), "cc": NS(p=0.3)}


def _patch_storage(monkeypatch, answers=None, mastery=MASTERY):
    from scholarpulse import storage
    monkeypatch.setattr(storage, "load_concepts", lambda: CONCEPTS)
    monkeypatch.setattr(storage, "load_answers", lambda sid: answers or [])
    monkeypatch.setattr(storage, "load_mastery", lambda sid: mastery)


class FakeLLM:
    """LLM 复核意见故意与确定性计算相反，验证不能覆盖工具结论。"""

    def chat_json(self, sys, user, **kw):
        return {"causal": False, "expected_delta": 0.99,
                "reasoning": "复核说无因果", "recommend_intervene": False}


def test_do_calculus_path_direction(monkeypatch):
    """因果路径必须是 干预→目标 方向（干预在上游源头，目标在末端）。"""
    _patch_storage(monkeypatch)
    monkeypatch.setattr(CA, "get_llm", lambda: FakeLLM())
    r = do_calculus.invoke({"sid": "s1", "target_cid": "cc", "intervention_cid": "ca"})
    assert r["causal_path"] == ["ca", "cb", "cc"]
    assert r["causal"] is True
    assert abs(r["expected_delta"] - 0.319) <= 0.001   # (1-0.15)×1/1.6×0.6


def test_do_calculus_llm_cannot_override(monkeypatch):
    _patch_storage(monkeypatch)
    monkeypatch.setattr(CA, "get_llm", lambda: FakeLLM())
    r = do_calculus.invoke({"sid": "s1", "target_cid": "cc", "intervention_cid": "ca"})
    assert r["causal"] is True                          # 确定性结论不被 LLM 覆盖
    assert abs(r["expected_delta"] - 0.319) <= 0.001
    assert r["llm_review"]["causal"] is False           # LLM 意见降级为参考字段
    assert r["llm_review"]["expected_delta"] == 0.99


def test_do_calculus_no_path_confounded(monkeypatch):
    """干预点不在目标上游（无因果路径）→ 混杂相关，效应 = 目标缺口×0.1。"""
    _patch_storage(monkeypatch)
    monkeypatch.setattr(CA, "get_llm", lambda: FakeLLM())
    r = do_calculus.invoke({"sid": "s1", "target_cid": "ca", "intervention_cid": "cc"})
    assert r["causal_path"] == [] and r["causal"] is False
    assert abs(r["expected_delta"] - 0.085) <= 0.001    # (1-0.15)×0.1


def test_build_causal_graph_temporal_and_spurious(monkeypatch):
    ans = [NS(cid="ca", ts=1, correct=True), NS(cid="cb", ts=2, correct=True),
           NS(cid="cc", ts=3, correct=True)]
    _patch_storage(monkeypatch, answers=ans)
    g = build_causal_graph.invoke({"sid": "s1"})
    assert [(e["src"], e["dst"]) for e in g["edges"]] == [("ca", "cb"), ("cb", "cc")]
    assert g["spurious_correlations"][0]["cid"] == "cb"  # B掌握但先修A未掌握→疑似死记硬背


def test_find_root_cause_walks_to_true_root(monkeypatch):
    """时序传导图下应溯源到最深薄弱根 A，而不是停在直接先修 B。"""
    ans = [NS(cid="ca", ts=1, correct=True), NS(cid="cb", ts=2, correct=True),
           NS(cid="cc", ts=3, correct=True)]
    _patch_storage(monkeypatch, answers=ans)
    r = find_root_cause.invoke({"sid": "s1", "target_cid": "cc"})
    assert r["root_cause"]["cid"] == "ca"
    assert r["cause_type"] == "知识缺口"                 # A 掌握度 0.15 < 0.2
    assert r["evidence_chain"][0]["node"] == "概念A"


def test_find_root_cause_structural_fallback(monkeypatch):
    """无答题时序 → 先修链兜底：直接阻塞 B 优先于被已掌握中介衰减的深层 A。"""
    _patch_storage(monkeypatch)
    r = find_root_cause.invoke({"sid": "s1", "target_cid": "cc"})
    assert r["root_cause"]["cid"] == "cb"
    assert r["cause_type"] == "技能薄弱"


def test_build_causal_graph_empty_guard(monkeypatch):
    _patch_storage(monkeypatch, answers=[], mastery={})
    g = build_causal_graph.invoke({"sid": "s1"})
    assert g["nodes"] == [] and "答题记录不足" in g["msg"]
