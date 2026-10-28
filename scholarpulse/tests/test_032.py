"""学区级多智能体治理（L0-L4）：人口合成 / ABM / 流行病学 / 政策沙盘 / 层级治理。"""
from scholarpulse import storage
from scholarpulse.society import abm, epidemic
from scholarpulse.society.population import seed_society
from scholarpulse.graph import hierarchy, policy_sandbox


# ---------- 1. L0 人口合成：可复现 + sim 前缀 ----------
def test_seed_reproducible_and_sim_prefix():
    info1 = seed_society(42)
    s1 = storage.load_society()
    info2 = seed_society(42)
    s2 = storage.load_society()
    assert info1 == info2 and info1["teachers"] == 80 and info1["students"] == 350
    # 同 seed 两次合成：特质/社交边/小组划分逐字段全等
    assert s1.traits == s2.traits and s1.edges == s2.edges and s1.groups == s2.groups
    # Windows 大小写不敏感：合成学生必须 sim 前缀，避免与真实学生 s001 撞 mastery 文件
    assert all(sid.startswith("sim") for sid in s2.traits)


# ---------- 2. L1 ABM：状态合法 ----------
def test_abm_states_legal():
    r = abm.run_days(5, 42)
    assert r["answers_generated"] > 0 and 0.0 < r["avg_mastery"] <= 1.0
    cids = {c.cid for c in storage.load_concepts()}
    soc = storage.load_society()
    # 不硬编码具体学生（高拖延个体可能 5 天零答题）：抽样全体验证状态合法性
    n_ans = 0
    for sid in soc.traits:
        assert all(0.05 <= cell.p <= 1.0 for cell in storage.load_mastery(sid).values())
        recs = storage.load_answers(sid)
        assert all(rec.cid in cids for rec in recs)
        n_ans += len(recs)
    assert n_ans == r["answers_generated"], "落盘答题总数与仿真统计不一致"


# ---------- 3. L3 流行病学：R0 对 β 单调 + 感染峰值单调 ----------
def test_r0_and_peak_monotonic():
    epidemic.propagate(30, 42)
    lo = {r.cid: r.r0 for r in epidemic.estimate_r0(beta=0.15)}
    hi = {r.cid: r.r0 for r in epidemic.estimate_r0(beta=0.35)}
    common = [cid for cid, v in lo.items() if v > 0]
    assert common, "无任何概念产生感染，数据异常"
    # R0 = β·k̄·D̄，k̄/D̄ 与 β 无关 ⇒ R0 对 β 严格递增
    assert all(hi[cid] > lo[cid] for cid in common)
    # 同 seed 两档 β：传播越快，感染峰值不降
    soc = epidemic._ensure_society()
    adj = epidemic._adjacency(soc)
    seeds = epidemic.init_states()
    assert seeds, "无自然感染种子（答题流为空？）"
    curve_lo = epidemic._run_sir(seeds, adj, soc.traits, 30, 42, beta=0.10)["curve"]
    curve_hi = epidemic._run_sir(seeds, adj, soc.traits, 30, 42, beta=0.40)["curve"]
    peak_lo = max(c["i"] for c in curve_lo)
    peak_hi = max(c["i"] for c in curve_hi)
    assert peak_hi >= peak_lo


# ---------- 3b. 幂迭代中心性：星形图枢纽最大 / 正则图退化均匀 ----------
def test_power_iteration_properties():
    star = {"hub": [(f"l{i}", 1.0) for i in range(5)]}
    star.update({f"l{i}": [("hub", 1.0)] for i in range(5)})
    _, eig = epidemic._power_iteration(star)
    assert eig["hub"] > max(eig[f"l{i}"] for i in range(5))
    # 6-循环（2-正则连通图）：主特征向量均匀退化
    cyc = {f"n{i}": [(f"n{(i + 1) % 6}", 1.0), (f"n{(i - 1) % 6}", 1.0)] for i in range(6)}
    _, eig2 = epidemic._power_iteration(cyc)
    vals = list(eig2.values())
    assert max(vals) - min(vals) < 1e-6


# ---------- 4. L4 政策沙盘：方向性（注入 policy_json 跳过 LLM） ----------
_STRONG = {"valid": True, "intent": "对掌握度低于0.4的学生每天增加2道复习题",
           "condition": {"avg_below": 0.4}, "action": "extra_review", "extra_questions": 2}


def test_sandbox_directionality():
    strong = policy_sandbox.run_sandbox("强干预", days=20, policy_json=_STRONG)["effect"]
    assert strong["delta_avg"] > 0, "强干预必须为正效应"
    lo, hi = strong["ci95"]
    assert lo <= strong["delta_avg"] <= hi, "bootstrap CI 必须包含点估计"
    placebo = policy_sandbox.run_sandbox(
        "安慰剂", days=20, policy_json=dict(_STRONG, extra_questions=0))["effect"]
    assert abs(placebo["delta_avg"]) < 0.02, "安慰剂效应必须趋近于零（平行世界逐位一致）"


# ---------- 5. L2 层级治理：Send 扇出 + LLM 降级 ----------
def test_governance_fanout_degrade(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("LLM down")
    monkeypatch.setattr(hierarchy, "get_llm", _boom)
    rep = hierarchy.run_governance(top_k=3)
    assert len(rep["briefs"]) == 80, "80 个班级画像必须收齐（幂等守卫）"
    deep = [b for b in rep["briefs"] if b["llm_analysis"]]
    assert len(deep) == 3, "只有 Top-K 进入深析"
    assert "LLM不可用" in deep[0]["llm_analysis"], "LLM 故障必须走规则降级"
    assert rep["global_report"] and rep["llm_calls"] == 4
