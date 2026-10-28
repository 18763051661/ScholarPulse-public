"""
【L1 ABM 行为仿真引擎】—— 教育系统的"物理引擎"
基于智能体的建模（Agent-Based Model）：350 个合成学生按各自潜在特质在虚拟天数里独立决策，
宏观模式（分层固化、两极分化、误区传染）从微观规则中涌现，而非预设。
每 tick（=1 虚拟天）每个学生依次经历：
  1. 艾宾浩斯遗忘 —— p *= exp(-forget_rate)，每天全体衰减，学习对抗遗忘
  2. 政策干预 —— 若反事实政策对该生生效：强制补弱复习（不受拖延影响，弱项优先）
  3. 拖延决策  —— 以 (1-λ) 概率才学习；拖延是分层固化的第一推手
  4. 选择性复习 —— 信心偏差决定看哪里：过度自信复习"已会的"（错位学习），过度不自信死磕"最弱的"
  5. 答题抽样  —— 对练习概念抛 Bernoulli(p)，产出真实答题记录
  6. 同伴误区传染 —— 沿社交边传播"错误认知"，受害者该概念掌握度受损
随机性设计（反事实归因的基石）：
  · 每个学生独立随机流 random.Random(f"{seed}:{sid}")——个体决策互不干扰
  · 误区传染用无状态 crc32 哈希流——某个体是否受干预不改变他人的随机数对齐
  · 因此"同 seed 平行世界"中对照组轨迹逐位一致，安慰剂对照严格干净（政策沙盘的前提）
纯规则、零 LLM。
"""
from __future__ import annotations
import math, random, time, zlib
from ..models import AnswerRecord, Concept, Mastery
from .. import storage
from .. import mastery as M
from .population import seed_society

FLOOR = 0.05          # 遗忘下限（不会彻底归零）
ATTEMPT_P = 0.55      # 每天尝试答题的概率
CONTAGION_BETA = 0.25 # 每次有效接触的误区传播概率
STUDY_K = 4           # 每天学习的概念数


def _study_targets(rng: random.Random, pmap: dict[str, float],
                   concepts: list[Concept], bias: str, k: int = STUDY_K) -> list[str]:
    """信心偏差决定复习目标的选择偏差——这是 ABM 里'心理'起作用的地方。"""
    weights = []
    for c in concepts:
        p = pmap.get(c.cid, M.PRIOR)
        if bias == "overconfident":
            w = p + 0.15                     # 错位学习：总复习已经会的
        elif bias == "underconfident":
            w = (1 - p) + 0.15               # 过度复习最弱的（其实健康的保守策略）
        else:
            w = c.exam_weight * (1 - p) + 0.1  # 平衡：薄弱 × 考权
        weights.append(max(w, 0.02))
    k = min(k, len(concepts))
    return [c.cid for c in rng.choices(concepts, weights=weights, k=k)]


def _cdraw(seed: int, day: int, src: str, dst: str, cid: str) -> float:
    """无状态确定性随机数（crc32 哈希流）：不依赖调用次序，干预不改变他人的随机数对齐。"""
    return zlib.crc32(f"{seed}|{day}|{src}|{dst}|{cid}".encode("utf-8")) / 4294967296.0


def _policy_applies(policy: dict, sid: str, pmap: dict[str, float]) -> bool:
    """政策是否作用于该生：须属于处理组（如指定 _treatment）且满足触发条件。"""
    treated = policy.get("_treatment")
    if treated is not None and sid not in treated:
        return False
    cond = policy.get("condition") or {}
    if cond.get("all"):
        return True
    thr = cond.get("avg_below")
    if thr is not None and pmap:
        return (sum(pmap.values()) / len(pmap)) < float(thr)
    return False


def _tick_day(state: dict, streams: dict[str, random.Random], day: int,
              concepts: list[Concept], adj: dict[str, list[tuple[str, float]]],
              seed: int = 42, policy: dict | None = None) -> int:
    """推进一天。state: {"mastery": {sid: {cid: p}}, "infected": {sid: set[cid]}}。返回当日答题数。"""
    mastery, infected = state["mastery"], state["infected"]
    n_answers = 0
    for sid, pmap in mastery.items():
        tr = state["traits"][sid]
        rng = streams[sid]
        today: list[AnswerRecord] = []
        # 1) 艾宾浩斯遗忘：全体衰减（学习与之对抗）
        for cid in list(pmap):
            pmap[cid] = max(FLOOR, pmap[cid] * math.exp(-tr.forget_rate))
        # 2) 政策干预（如生效）：额外复习为强制任务，弱项优先。
        #    此分支不消耗随机数——安慰剂（强度0）下两个世界逐位一致
        extra_cids: list[str] = []
        if policy is not None and _policy_applies(policy, sid, pmap):
            k_extra = int(policy.get("extra_questions", 0))
            if k_extra > 0:
                extra_cids = [cid for cid, _ in
                              sorted(pmap.items(), key=lambda kv: (kv[1], kv[0]))[:k_extra]]
                gain = M.GAIN * (1.0 + 0.8 * tr.ability_prior)
                for cid in extra_cids:
                    pmap[cid] = min(1.0, pmap[cid] + (1 - pmap[cid]) * gain)
        # 3) 拖延决策（只影响自主学习）
        if rng.random() < tr.procrastination:
            procrastinated = True
        else:
            procrastinated = False
        # 4) 选择性复习 + 学习增量（效率随能力）
        if not procrastinated:
            targets = _study_targets(rng, pmap, concepts, tr.confidence_bias)
            gain = M.GAIN * (1.0 + 0.8 * tr.ability_prior)
            for cid in targets:
                pmap[cid] = min(1.0, pmap[cid] + (1 - pmap[cid]) * gain)
            # 5) 答题抽样（产出真实记录，喂给后续所有分析层）
            if rng.random() < ATTEMPT_P:
                quiz_cids = rng.sample(targets, 1) + rng.sample(list(pmap), 1) if len(pmap) > 1 else targets
                for cid in set(quiz_cids):
                    correct = rng.random() < pmap.get(cid, M.PRIOR)
                    pmap[cid] = M.apply_answer(MasteryShim(cid, pmap[cid]), correct).p
                    rec = AnswerRecord(qid=f"sim_d{day:02d}_{sid}_{cid}", cid=cid,
                                       correct=correct, ts=f"virtual-day-{day:02d}")
                    state["answers"][sid].append(rec)
                    today.append(rec)
                    n_answers += 1
        # 5b) 政策强制复习题作答（必做作业，同样走贝叶斯更新）
        for cid in extra_cids:
            correct = rng.random() < pmap.get(cid, M.PRIOR)
            pmap[cid] = M.apply_answer(MasteryShim(cid, pmap[cid]), correct).p
            rec = AnswerRecord(qid=f"poly_d{day:02d}_{sid}_{cid}", cid=cid,
                               correct=correct, ts=f"virtual-day-{day:02d}")
            state["answers"][sid].append(rec)
            today.append(rec)
            n_answers += 1
        # 6) 误区状态更新（当日新记录）：答错 → 感染该概念误区；答对 → 康复
        for rec in today:
            if rec.correct:
                infected[sid].discard(rec.cid)
            elif rec.cid not in infected[sid] and rng.random() < 0.5:
                infected[sid].add(rec.cid)
    # 7) 同伴误区传染：沿社交边，感染者的邻居以 beta·w·social_activity 概率被击穿
    for src, neighbors in adj.items():
        for cid in list(infected[src]):
            for dst, w in neighbors:
                if cid in infected[dst] or cid not in mastery[dst]:
                    continue
                p_hit = CONTAGION_BETA * w * state["traits"][dst].social_activity
                if _cdraw(seed, day, src, dst, cid) < p_hit:
                    mastery[dst][cid] *= 0.9      # 被传染：掌握度受损
                    infected[dst].add(cid)
    return n_answers


class MasteryShim:
    """轻量掌握度替身：复用 mastery.apply_answer 的收敛规则而不落盘。"""
    def __init__(self, cid: str, p: float):
        self.cid, self.p = cid, p
        self.n_correct = 0
        self.n_wrong = 0
        self.updated_at = ""


def _ensure_society():
    soc = storage.load_society()
    if soc is None:
        seed_society(42)
        soc = storage.load_society()
    return soc


def _init_state(sim_sids: list[str], concepts: list[Concept], soc) -> dict:
    """初始状态：能力先验决定的异质起点（分层从第一天就存在）。"""
    mastery = {sid: {c.cid: max(FLOOR, min(1.0, M.PRIOR + (soc.traits[sid].ability_prior - 0.5) * 0.3))
                     for c in concepts if c.noise != "filtered"} for sid in sim_sids}
    return {"mastery": mastery, "traits": soc.traits,
            "infected": {sid: set() for sid in sim_sids},
            "answers": {sid: [] for sid in sim_sids}}


def _build_adj(soc, sim_sids: list[str]) -> dict[str, list[tuple[str, float]]]:
    """邻接表（无向，双向传播）。"""
    adj: dict[str, list[tuple[str, float]]] = {sid: [] for sid in sim_sids}
    for e in soc.edges:
        if e.src in adj and e.dst in adj:
            adj[e.src].append((e.dst, e.weight))
            adj[e.dst].append((e.src, e.weight))
    return adj


def _simulate(state: dict, concepts: list[Concept], adj: dict[str, list[tuple[str, float]]],
              days: int, seed: int = 42, policy: dict | None = None,
              track_sids: list[str] | None = None) -> tuple[list, list]:
    """纯内存推进 days 天（不落盘）。返回 (每日曲线, track_sids 每日平均掌握度)。"""
    streams = {sid: random.Random(f"{seed}:{sid}") for sid in state["mastery"]}
    track_set = set(track_sids or ())
    curve, track = [], []
    for d in range(1, days + 1):
        n = _tick_day(state, streams, d, concepts, adj, seed, policy)
        total = sum(len(s) for s in state["infected"].values())
        curve.append({"day": d, "answers": n, "infected_cells": total})
        if track_set:
            vals = [sum(pm.values()) / len(pm) for sid, pm in state["mastery"].items()
                    if sid in track_set]
            track.append(round(sum(vals) / len(vals), 6) if vals else 0.0)
    return curve, track


def run_world(days: int = 30, seed: int = 42, policy: dict | None = None,
              treatment: list[str] | None = None, track_sids: list[str] | None = None) -> dict:
    """纯内存反事实世界（不落盘）：政策沙盘的双跑引擎。policy 为结构化干预参数。"""
    soc = _ensure_society()
    concepts = storage.load_concepts()
    sim_sids = sorted(soc.traits)                      # 只仿真合成人口，不碰真实学生
    state = _init_state(sim_sids, concepts, soc)
    adj = _build_adj(soc, sim_sids)
    pol = None
    if policy:
        pol = dict(policy)
        pol["_treatment"] = set(treatment or ())
    curve, track = _simulate(state, concepts, adj, days, seed, pol, track_sids)
    return {"avg": {sid: round(sum(pm.values()) / len(pm), 6) for sid, pm in state["mastery"].items()},
            "curve": curve, "track": track}


def run_days(days: int = 30, seed: int = 42) -> dict:
    """仿真 days 个虚拟天并把 350 名合成学生的终态批量落盘。同 seed 结果可复现。"""
    t0 = time.time()
    soc = _ensure_society()
    concepts = storage.load_concepts()
    students = storage.load_students()
    sim_sids = sorted(soc.traits)                      # 只仿真合成人口，不碰真实学生
    state = _init_state(sim_sids, concepts, soc)
    adj = _build_adj(soc, sim_sids)
    sir_curve, _ = _simulate(state, concepts, adj, days, seed)
    # 终态批量落盘：700 个小文件各写一次（整写，非读改写）
    mastery = state["mastery"]
    n_answers_total = 0
    for sid in sim_sids:
        pmap = mastery[sid]
        records = state["answers"][sid]
        n_answers_total += len(records)
        mdict = {}
        for cid, p in pmap.items():
            m = MasteryShim(cid, p)
            m.n_correct = sum(1 for r in records if r.cid == cid and r.correct)
            m.n_wrong = sum(1 for r in records if r.cid == cid and not r.correct)
            m.updated_at = storage.now_iso()
            mdict[cid] = m
        storage.save_mastery(sid, {cid: _to_mastery(m) for cid, m in mdict.items()})
        storage.save_answers(sid, records)
        avg = M.avg_mastery(mdict)
        st = students.get(sid)
        if st:                                          # 同步层级，教师端可见
            st.tier = M.student_tier(avg)
    storage.save_students(students)
    # 仿真元数据（供疫情层复用 SIR 曲线）
    epi = storage.load_epidemic()
    epi["sir_curve"] = sir_curve
    epi["meta"] = {"days": days, "seed": seed, "answers_generated": n_answers_total}
    storage.save_epidemic(epi)
    all_p = [p for pmap in mastery.values() for p in pmap.values()]
    return {"days": days, "seed": seed, "answers_generated": n_answers_total,
            "avg_mastery": round(sum(all_p) / len(all_p), 4),
            "runtime_ms": int((time.time() - t0) * 1000)}


def _to_mastery(shim: MasteryShim) -> Mastery:
    return Mastery(cid=shim.cid, p=shim.p, n_correct=shim.n_correct,
                   n_wrong=shim.n_wrong, updated_at=shim.updated_at)
