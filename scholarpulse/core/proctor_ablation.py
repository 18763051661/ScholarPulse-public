"""监考裁决消融实验：单维规则（升级前） vs 多维交叉验证（升级后）量化对比。

回答一个问题：『为什么必须多维交叉验证，而不是单维异常就报警？』
用数据说话——同一批合成考试会话（30 正常 + 10 作弊，固定种子完全可复现），
分别用两套裁决器判定，对比误报率/漏报率/精确率/F1：

  升级前（单维规则）：任一维度异常即判作弊 —— 节奏快/相似度高/行为模式/
                      摄像头异常/音频异常/逐帧视觉异常，撞到一条就报。
  升级后（多维交叉）：复用生产环境真实工具链 scan_timing/scan_similarity/
                      scan_pattern/scan_camera/scan_frame_vision/scan_audio →
                      cross_validate 交叉验证（多维度同向才算强证据链）+
                      patrol_timeline 巡逻时序（跨源异常关联），
                      只有交叉验证命中或巡逻发现高危链条才判作弊。

作弊手法 4 类（每份会话 ≥2 个维度同向异常）：
  背答案(快答+高相似) / 中途换人(画面剧变+节奏突变) /
  朗读答案(突发高音+碎片作答) / 离席查资料(长时间无人+批量补填)。

正常考生含单维噪声（偶发快答/一次高音/低头遮挡帧）——这正是单维规则误报的来源。

全程离线零 LLM 调用，跑一次 <1 秒；`python -m scholarpulse.core.proctor_ablation` 查看报告。
"""
from __future__ import annotations

import json
import random

from ..agents.exam_proctor_agent import (
    cross_validate,
    patrol_timeline,
    scan_audio,
    scan_camera,
    scan_frame_vision,
    scan_pattern,
    scan_similarity,
    scan_timing,
)

N_FRAMES = 36          # 摄像头/麦克风采样数（前端每 5 秒一次，约 3 分钟）
SHOTS = list(range(0, N_FRAMES * 5, 5))

# 10 道题的（参考答案, 正常考生作答）模板——字面低重叠，保证正常相似度不越阈
QA_POOL = [
    ("临界区互斥：一次仅允许一个进程进入", "我用信号量来协调并发访问，避免资源竞争"),
    ("时间片轮转调度的核心是公平分配CPU", "系统按固定间隔轮流执行各个就绪任务"),
    ("死锁四要素包括占有等待与循环等待", "多个任务互相卡住资源谁也不肯让步"),
    ("虚拟内存通过页面置换扩展地址空间", "把暂时不用的数据换出去腾出运行空间"),
    ("TCP三次握手确保连接双方准备就绪", "建立网络连接前要先来回确认三个回合"),
    ("哈希表以键值对形式实现快速查找", "根据关键字直接算出存放位置来检索"),
    ("二叉树中序遍历先访问左子树再根节点", "按从小到大的顺序把节点逐个读出来"),
    ("事务ACID特性保证数据库一致性", "操作要么全部成功要么全部回滚"),
    ("贪心算法每步取局部最优未必全局最优", "只顾眼前选择的策略不一定拿到最好结果"),
    ("快速排序平均时间复杂度为nlogn", "分而治之的排序方法整体效率较高"),
]

CHEAT_KINDS = ["背答案", "中途换人", "朗读答案", "离席查资料"]


# ---------------------------------------------------------------- 合成会话
def _answer_for(ref: str, rng: random.Random) -> str:
    """正常作答：从 QA_POOL 取与 ref 配对的话术，首尾加自然抖动（长度差异化）。"""
    ans = next(a for r, a in QA_POOL if r == ref)
    prefix = rng.choice(["", "我觉得", "老师讲过", ""])
    suffix = rng.choice(["", "，大致如此", "，这是我的理解", "。", "，请老师指正", "，应该没错"])
    return prefix + ans + suffix


def _paraphrase(ref: str, rng: random.Random) -> str:
    """作弊者的『高级伪装』作答：换人/离席场景下答案本身正常（内容不露馅），
    露馅的是行为维度——这正是单维内容审查抓不到、多维交叉才有效的原因。"""
    return _answer_for(ref, rng)


def _frames(kind: str, rng: random.Random) -> list[dict]:
    """生成逐帧视觉特征 [{t,bright,motion,absent,motion_high,leave}]。"""
    frames = []
    for t in SHOTS:
        fr = {"t": t, "bright": rng.randint(80, 110), "motion": round(rng.uniform(0.02, 0.08), 2),
              "absent": False, "motion_high": False, "leave": False}
        if kind == "中途换人" and t % 10 < 5:          # 一半帧画面剧变
            fr.update(motion=round(rng.uniform(0.4, 0.7), 2), motion_high=True)
        if kind == "离席查资料" and t % 10 >= 5:       # 一半帧无人
            fr.update(bright=rng.randint(3, 8), motion=0.0, absent=True, leave=True)
        frames.append(fr)
    if kind == "正常低头噪声":                          # 1-2 帧低头（亮度偏低但未无人）
        for t in rng.sample(SHOTS, rng.randint(1, 2)):
            frames[t // 5].update(bright=rng.randint(25, 35))
    return frames


def _session(kind: str, rng: random.Random) -> dict:
    """生成一场合成考试会话：submit_log + camera + audio + events + label。"""
    refs = [r for r, _ in QA_POOL]
    # ---- 答题节奏 ----
    if kind == "背答案":
        durs = [rng.randint(2, 4) for _ in refs]               # 秒答
    elif kind == "中途换人":
        durs = [rng.randint(560, 660), rng.randint(560, 660)] + [rng.randint(3, 5)] * 8
        rng.shuffle(durs)                                       # 前后段节奏断裂
    elif kind == "离席查资料":
        durs = [0] * 6 + [rng.randint(85, 115) for _ in range(4)]  # 跳题批量补填
    else:
        durs = [rng.randint(45, 120) for _ in refs]
        if kind == "正常快答噪声":
            durs[rng.randrange(len(durs))] = rng.randint(8, 12)  # 偶发秒答一道

    # ---- 作答内容 ----
    if kind == "背答案":
        anss = list(refs)                                       # 与参考答案逐字相同
    else:
        anss = [_answer_for(r, rng) for r in refs]
        # 保底：随机一题带长后缀，保证作答长度方差超过"模板式"阈值（真实学生详略不一）
        anss[rng.randrange(len(anss))] += rng.choice(
            ["，以上是我结合课件内容的理解", "，这一点老师上课重点强调过"])
    if kind == "朗读答案":
        anss = [a[: rng.randint(3, 6)] for a in anss]           # 只写关键词（碎片式）

    submit_log = [{"i": i, "dur": d, "ans": a, "ref": r}
                  for i, (d, a, r) in enumerate(zip(durs, anss, refs))]

    # ---- 摄像头 ----
    frames = _frames("正常低头噪声" if kind == "正常低头噪声" else kind, rng)
    n_absent = sum(1 for f in frames if f["absent"])
    n_motion = sum(1 for f in frames if f["motion_high"])
    cam = {"authorized": True, "shots": N_FRAMES,
           "absent_frames": n_absent, "motion_high_frames": n_motion,
           "leave_events": 2 if kind == "离席查资料" else 0,
           "leave_log": [70, 140] if kind == "离席查资料" else [],
           "avg_brightness": round(sum(f["bright"] for f in frames) / N_FRAMES, 1),
           "frames_count": 2 if kind != "正常" else 0,
           "frame_log": frames}

    # ---- 麦克风 ----
    loud = 12 if kind == "朗读答案" else (rng.randint(5, 6) if kind == "正常高音噪声" else rng.randint(0, 1))
    aud = {"authorized": True, "samples": N_FRAMES,
           "voice_frames": rng.randint(6, 10), "loud_frames": loud,
           "avg_rms": round(0.03 + 0.01 * loud / 12, 3)}

    # ---- 巡逻事件（离席/换人作弊 + 交卷事件）----
    events = []
    if kind == "离席查资料":
        events = [{"t": 70, "source": "frame", "type": "absent", "desc": "考生离开画面"},
                  {"t": 150, "source": "answer", "type": "fast", "desc": "批量快速提交"},
                  {"t": 140, "source": "frame", "type": "absent", "desc": "再次离开"}]
    elif kind == "中途换人":
        events = [{"t": 200, "source": "frame", "type": "motion_high", "desc": "画面剧烈变化"},
                  {"t": 210, "source": "answer", "type": "break", "desc": "节奏突变"},
                  {"t": 320, "source": "frame", "type": "motion_high", "desc": "画面再次剧变"}]
    elif kind == "朗读答案":
        events = [{"t": 90, "source": "audio", "type": "loud", "desc": "突发高音"}]

    label = "正常" if kind.startswith("正常") else kind
    return {"kind": kind, "label": label,
            "submit_log": submit_log, "camera": cam, "audio": aud, "events": events}


def build_dataset(seed: int = 42) -> list[dict]:
    """固定种子合成 40 份会话：18 纯净正常 + 12 单维噪声正常 + 10 作弊（4 类手法）。"""
    rng = random.Random(seed)
    sessions = [_session("正常", rng) for _ in range(18)]
    # 单维噪声正常考生：4 快答 + 4 高音 + 4 低头（每种只在一个维度越阈）
    sessions += [_session("正常快答噪声", rng) for _ in range(4)]
    sessions += [_session("正常高音噪声", rng) for _ in range(4)]
    sessions += [_session("正常低头噪声", rng) for _ in range(4)]
    # 作弊 10 份：背答案3 + 换人2 + 朗读2 + 离席3
    sessions += [_session("背答案", rng) for _ in range(3)]
    sessions += [_session("中途换人", rng) for _ in range(2)]
    sessions += [_session("朗读答案", rng) for _ in range(2)]
    sessions += [_session("离席查资料", rng) for _ in range(3)]
    return sessions


# ---------------------------------------------------------------- 两套裁决器
def judge(session: dict) -> dict:
    """对一场会话分别跑『单维规则』与『多维交叉』两套裁决器。

    单维规则（升级前）：任一维度异常即判作弊。
    多维交叉（升级后）：cross_validate 交叉验证命中强证据链，
                        或 patrol_timeline 巡逻发现高危链条，才判作弊。
    """
    sl = json.dumps(session["submit_log"], ensure_ascii=False)
    tim = scan_timing.invoke({"submit_log": sl})
    sim = scan_similarity.invoke({"submit_log": sl})
    pat = scan_pattern.invoke({"submit_log": sl})
    cam = scan_camera.invoke({"camera_json": json.dumps(session["camera"], ensure_ascii=False)})
    aud = scan_audio.invoke({"audio_json": json.dumps(session["audio"], ensure_ascii=False)})
    fv = scan_frame_vision.invoke(
        {"frame_log": json.dumps(session["camera"]["frame_log"], ensure_ascii=False)})
    cv = cross_validate.invoke({"evidence_json": json.dumps(
        {"timing": tim, "similarity": sim, "pattern": pat,
         "camera": cam, "audio": aud}, ensure_ascii=False)})
    pt = (patrol_timeline.invoke({"events_json": json.dumps(session["events"], ensure_ascii=False)})
          if session["events"] else {"patrol_verdict": "no_events"})

    single_flags = []
    if tim.get("fast_questions"):
        single_flags.append("节奏快答")
    if tim.get("rhythm_breaks"):
        single_flags.append("节奏突变")
    if sim.get("high_similarity"):
        single_flags.append("高相似")
    if pat.get("pattern") != "正常作答":
        single_flags.append(pat["pattern"])
    if cam.get("camera_flags"):
        single_flags += cam["camera_flags"]
    if aud.get("audio_flags"):
        single_flags += aud["audio_flags"]
    if fv.get("n_abnormal", 0) > 0:
        single_flags.append(f"逐帧视觉异常{fv['n_abnormal']}帧")

    baseline_flag = bool(single_flags)
    upgraded_flag = (bool(cv.get("correlated_risks"))
                     or pt.get("patrol_verdict") == "suspected_cheating")
    return {"kind": session["kind"], "label": session["label"],
            "baseline_flag": baseline_flag, "upgraded_flag": upgraded_flag,
            "single_flags": single_flags,
            "correlated": [r["risk"] for r in cv.get("correlated_risks", [])],
            "confidence": cv.get("confidence", 0),
            "patrol_verdict": pt.get("patrol_verdict", "no_events")}


def _rates(rows: list[dict], key: str) -> dict:
    tp = sum(1 for r in rows if r["label"] != "正常" and r[key])
    fp = sum(1 for r in rows if r["label"] == "正常" and r[key])
    tn = sum(1 for r in rows if r["label"] == "正常" and not r[key])
    fn = sum(1 for r in rows if r["label"] != "正常" and not r[key])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fpr": round(fp / (fp + tn), 3) if fp + tn else 0.0,
            "fnr": round(fn / (fn + tp), 3) if fn + tp else 0.0,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def run_experiment(seed: int = 42) -> dict:
    """跑完整消融实验，返回两套裁决器的混淆矩阵与量化指标（含逐案明细）。"""
    rows = [judge(s) for s in build_dataset(seed)]
    by_kind = {}
    for r in rows:
        if r["label"] != "正常":
            by_kind.setdefault(r["label"], {"n": 0, "caught_baseline": 0, "caught_upgraded": 0})
            by_kind[r["label"]]["n"] += 1
            by_kind[r["label"]]["caught_baseline"] += int(r["baseline_flag"])
            by_kind[r["label"]]["caught_upgraded"] += int(r["upgraded_flag"])
    base, up = _rates(rows, "baseline_flag"), _rates(rows, "upgraded_flag")
    return {"dataset": {"normal": sum(1 for r in rows if r["label"] == "正常"),
                        "cheating": sum(1 for r in rows if r["label"] != "正常"),
                        "seed": seed},
            "baseline_single_dim": base,
            "upgraded_cross_validated": up,
            "cheat_recall_by_kind": by_kind,
            "headline": (f"误报率 {base['fpr']:.0%} → {up['fpr']:.0%}（降 {base['fpr'] - up['fpr']:.0%}），"
                         f"漏报率同为 {base['fnr']:.0%}；F1 {base['f1']:.2f} → {up['f1']:.2f}"),
            "cases": rows}


if __name__ == "__main__":
    rep = run_experiment()
    print("=" * 62)
    print("监考裁决消融实验：单维规则(升级前) vs 多维交叉验证(升级后)")
    print("=" * 62)
    for name, m in (("升级前·单维规则", rep["baseline_single_dim"]),
                    ("升级后·多维交叉", rep["upgraded_cross_validated"])):
        print(f"\n[{name}]  TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
        print(f"  误报率(FPR)={m['fpr']:.1%}  漏报率(FNR)={m['fnr']:.1%}  "
              f"精确率={m['precision']:.1%}  F1={m['f1']:.2f}")
    print("\n各作弊手法捕获情况（baseline/upgraded）:")
    for k, v in rep["cheat_recall_by_kind"].items():
        print(f"  {k}: {v['caught_baseline']}/{v['n']} vs {v['caught_upgraded']}/{v['n']}")
    print(f"\n结论：{rep['headline']}")
