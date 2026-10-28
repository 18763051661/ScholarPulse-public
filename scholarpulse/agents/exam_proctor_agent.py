"""在线考试 AI 防作弊监考官（LangChain ReAct 智能体）。

博士级创新点：把考试防作弊从『人工巡查+简单规则』升级为
『LLM ReAct 智能体多维度取证 → 交叉验证强证据链 → 成绩合理性预测 → 自主裁决风险等级 → 出具监考报告』。

监考官可调用 8 个真实数据取证工具：
  1. scan_timing              — 答题节奏异常检测（过快=背答案/过慢=查资料/突变=换人）
  2. scan_similarity          — 作答与参考答案相似度（逐题字面重合+关键词覆盖）
  3. scan_pattern             — 作答行为模式（突发式/碎片式/异常停顿/前后矛盾）
  4. scan_camera              — 摄像头视觉取证（前端每5秒抓帧：无人率/离开次数/画面剧变）
  5. scan_audio               — 麦克风语音取证（无声率/突发高音/多人喧哗嫌疑）
  6. cross_validate           — 多维证据交叉验证（五维关联→强证据链→降误报）
  7. predict_score_plausibility — 成绩合理性预测（对比历史正确率，异常高/低）
  8. issue_verdict            — 出具监考裁决书（唯一落盘入口）

摄像头链路：浏览器 getUserMedia 实时视频流 → 前端逐帧灰度化分析
（亮度=无人检测、帧差=画面剧变检测）→ 视觉统计随交卷上送 →
本智能体 scan_camera 工具量化取证 → cross_validate 交叉验证 → LLM API 综合裁决。

裁决维度：节奏异常 / 内容抄袭 / 行为模式 / 摄像头视觉 / 音频语音 /
交叉验证强证据链 / 成绩合理性 / 综合风险，并给出处置建议。
"""
from __future__ import annotations
from langchain_core.tools import tool
from .diagnosis_agents import _run_react, _strip_markdown
from ..lc_model import RouterChatModel


@tool
def scan_timing(submit_log: str) -> dict:
    """扫描答题节奏异常：逐题耗时/平均耗时/最短最长/突变点。

    submit_log 为 JSON 字符串，形如 [{"i":0,"dur":120,"ans":"..."}, ...]，
    dur 单位为秒。返回节奏指标 + 异常标记。"""
    import json
    try:
        log = json.loads(submit_log)
    except Exception:
        return {"error": "submit_log 非合法 JSON"}
    if not isinstance(log, list) or not log:
        return {"n": 0, "msg": "无作答记录"}
    durs = [x.get("dur", 0) for x in log if isinstance(x, dict)]
    if not durs:
        return {"n": 0, "msg": "无耗时数据"}
    n = len(durs)
    avg = sum(durs) / n
    mn, mx = min(durs), max(durs)
    # 异常：某题耗时 < 平均的 1/4（秒答）或 > 平均的 4 倍（久停）
    fast = [i for i, d in enumerate(durs) if d < max(avg * 0.25, 3)]
    slow = [i for i, d in enumerate(durs) if d > avg * 4 and d > 30]
    # 节奏突变：相邻题耗时差超过 3 倍平均
    breaks = [i for i in range(1, n)
              if abs(durs[i] - durs[i - 1]) > avg * 3]
    # 前后段均值对比（判断是否中途换策略）
    half = max(n // 2, 1)
    fa = sum(durs[:half]) / half
    ba = sum(durs[half:]) / max(n - half, 1)
    shift = "前快后慢" if fa < ba * 0.5 else "前慢后快" if ba < fa * 0.5 else "平稳"
    return {"n": n, "avg_sec": round(avg, 1), "min_sec": mn, "max_sec": mx,
            "fast_questions": fast[:5], "slow_questions": slow[:5],
            "rhythm_breaks": breaks[:5], "pace_shift": shift}


@tool
def scan_similarity(submit_log: str) -> dict:
    """扫描作答与参考答案的相似度：逐题字面重合率+关键词覆盖+疑似抄写题号。

    submit_log 为 JSON 字符串，形如 [{"i":0,"ans":"...","ref":"..."}, ...]。
    相似度 > 0.85 且耗时极短 → 疑似背答案/抄写。"""
    import json, re
    try:
        log = json.loads(submit_log)
    except Exception:
        return {"error": "submit_log 非合法 JSON"}
    if not isinstance(log, list) or not log:
        return {"n": 0, "msg": "无作答记录"}
    suspicious = []
    sims = []
    for x in log:
        if not isinstance(x, dict):
            continue
        ans = (x.get("ans", "") or "").strip()
        ref = (x.get("ref", "") or "").strip()
        if not ans or not ref:
            sims.append(0.0)
            continue
        # 字符级 Jaccard：去标点后按字符集比较
        a_set = set(re.sub(r"[，。、,.！!？?；;：:\s]+", "", ans))
        r_set = set(re.sub(r"[，。、,.！!？?；;：:\s]+", "", ref))
        jacc = len(a_set & r_set) / max(len(a_set | r_set), 1)
        # 词覆盖：参考答案的关键词被作答覆盖的比例
        r_words = set(ref) - set("，。、,.！！？？；;：: \n")
        cover = len(r_words & set(ans)) / max(len(r_words), 1)
        sim = round(max(jacc, cover), 2)
        sims.append(sim)
        if sim > 0.85:
            suspicious.append(x.get("i", "?"))
    avg_sim = round(sum(sims) / max(len(sims), 1), 2)
    return {"n": len(sims), "avg_sim": avg_sim,
            "max_sim": max(sims) if sims else 0,
            "suspicious_questions": suspicious[:8],
            "high_similarity": avg_sim > 0.75}


@tool
def scan_pattern(submit_log: str) -> dict:
    """扫描作答行为模式：突发式/碎片式/异常停顿/作答长度分布。

    突发式=全部在极短时间内连续提交（疑似换人代答）；
    碎片式=每题作答极短且雷同（疑似关键词堆砌）。"""
    import json
    try:
        log = json.loads(submit_log)
    except Exception:
        return {"error": "submit_log 非合法 JSON"}
    if not isinstance(log, list) or not log:
        return {"n": 0, "msg": "无作答记录"}
    durs = [x.get("dur", 0) for x in log if isinstance(x, dict)]
    anss = [x.get("ans", "") for x in log if isinstance(x, dict)]
    lens = [len(a) for a in anss]
    n = len(durs)
    if not n:
        return {"n": 0}
    avg_dur = sum(durs) / n
    avg_len = sum(lens) / max(len(lens), 1)
    # 突发式：总耗时极短但题数多
    burst = sum(durs) < avg_dur * n * 0.3 and n >= 5
    # 碎片式：作答都极短
    fragment = avg_len < 8 and n >= 5
    # 长度方差：全部一样长（疑似模板填充）
    import statistics
    len_std = statistics.pstdev(lens) if len(lens) > 1 else 0
    templated = len_std < 2 and avg_len > 5 and n >= 4
    # 停顿：连续多题耗时为 0（跳题后批量补）
    zeros = sum(1 for d in durs if d == 0)
    batch_fill = zeros > n * 0.5
    pattern = ("突发式(疑似代答)" if burst
               else "碎片式(疑似堆砌)" if fragment
               else "模板式(疑似填充)" if templated
               else "批量补填(疑似跳题)" if batch_fill
               else "正常作答")
    return {"n": n, "avg_dur_sec": round(avg_dur, 1), "avg_len": round(avg_len, 1),
            "len_std": round(len_std, 1), "n_zero_dur": zeros,
            "burst": burst, "fragment": fragment, "templated": templated,
            "batch_fill": batch_fill, "pattern": pattern}


@tool
def scan_camera(camera_json: str) -> dict:
    """摄像头视觉取证：分析考试期间的摄像头抓帧统计（前端每5秒抓帧分析）。

    camera_json 为 JSON 字符串：{"authorized":true,"shots":30,"absent_frames":4,
    "motion_high_frames":2,"leave_events":1,"leave_log":[12],"avg_brightness":86.5,
    "frames_count":3}。
    无人率>30% → 疑似离席；离开≥2次 → 多次离开；剧变帧占比>40% → 疑似多人/换人；
    拒绝授权或零抓帧 → 拒绝监考。返回量化指标 + camera_flags 异常标记。"""
    import json
    try:
        cam = json.loads(camera_json)
    except Exception:
        return {"error": "camera_json 非合法 JSON"}
    if not isinstance(cam, dict) or not cam:
        return {"error": "无摄像头数据"}
    if not cam.get("authorized", False):
        return {"authorized": False,
                "camera_flags": ["拒绝摄像头授权"],
                "note": f"考生拒绝摄像头授权（{cam.get('error', '未知原因')}），全程无视觉证据"}
    shots = int(cam.get("shots", 0))
    if shots == 0:
        return {"authorized": True, "shots": 0,
                "camera_flags": ["摄像头零抓帧"],
                "note": "已授权但无任何抓帧，摄像头流可能被系统挂起或人为遮挡"}
    absent = int(cam.get("absent_frames", 0))
    motion_high = int(cam.get("motion_high_frames", 0))
    leave_events = int(cam.get("leave_events", 0))
    leave_log = cam.get("leave_log", [])
    avg_bright = float(cam.get("avg_brightness", 0))
    absence_rate = round(absent / shots, 2)
    motion_rate = round(motion_high / shots, 2)
    flags = []
    if absence_rate > 0.3:
        flags.append("疑似离席(无人率>30%)")
    if leave_events >= 2:
        flags.append("多次离开画面")
    if motion_rate > 0.4:
        flags.append("画面剧烈变化(多人/换人嫌疑)")
    if avg_bright < 10:
        flags.append("画面持续过暗(镜头遮挡嫌疑)")
    frames_count = int(cam.get("frames_count", 0))
    return {"authorized": True, "shots": shots,
            "absence_rate": absence_rate, "motion_rate": motion_rate,
            "leave_events": leave_events, "leave_log": leave_log[:8],
            "avg_brightness": avg_bright, "frames_count": frames_count,
            "camera_flags": flags,
            "note": ("视觉正常" if not flags
                     else f"视觉异常{len(flags)}项，建议结合节奏/相似度交叉验证")
                   + (f"，关键帧存证{frames_count}张" if frames_count else "")}


@tool
def scan_audio(audio_json: str) -> dict:
    """麦克风语音取证：分析考试期间的考场音频统计（前端每5秒采样RMS音量）。

    audio_json 为 JSON 字符串：{"authorized":true,"samples":30,"voice_frames":12,
    "loud_frames":1,"avg_rms":0.045}。
    voice_rate<4% → 考场持续无声(疑似离席/静音作弊)；loud_rate>25% →
    多次突发高音(疑似喧哗/多人讨论/朗读答案)。返回量化指标 + audio_flags 异常标记。"""
    import json
    try:
        aud = json.loads(audio_json)
    except Exception:
        return {"error": "audio_json 非合法 JSON"}
    if not isinstance(aud, dict) or not aud:
        return {"error": "无音频数据"}
    if not aud.get("authorized", False):
        return {"authorized": False,
                "audio_flags": ["麦克风未授权"],
                "note": f"考生未授权麦克风（{aud.get('error', '未知原因')}），全程无音频证据"}
    samples = int(aud.get("samples", 0))
    if samples == 0:
        return {"authorized": True, "samples": 0,
                "audio_flags": ["麦克风零采样"],
                "note": "已授权但无任何采样，音频流可能被系统挂起"}
    voice = int(aud.get("voice_frames", 0))
    loud = int(aud.get("loud_frames", 0))
    avg_rms = float(aud.get("avg_rms", 0))
    voice_rate = round(voice / samples, 2)
    loud_rate = round(loud / samples, 2)
    flags = []
    if voice_rate < 0.04:
        flags.append("考场持续无声(疑似离席/静音作弊)")
    if loud_rate > 0.25:
        flags.append("多次突发高音(疑似喧哗/多人讨论)")
    elif loud_rate > 0.1:
        flags.append("存在明显高音段(需结合视觉交叉验证)")
    return {"authorized": True, "samples": samples,
            "voice_rate": voice_rate, "loud_rate": loud_rate,
            "avg_rms": round(avg_rms, 4),
            "audio_flags": flags,
            "note": ("音频正常" if not flags
                     else f"音频异常{len(flags)}项，建议与摄像头视觉证据交叉验证")}


@tool
def cross_validate(evidence_json: str) -> dict:
    """多维证据交叉验证：把节奏/相似度/行为/摄像头/音频五维证据做关联分析，
    找出『多维度同时指向同一作弊手段』的强证据链。

    evidence_json 为 JSON 字符串：{"timing":{...},"similarity":{...},"pattern":{...},
    "camera":{...},"audio":{...}}（各维 scan_* 工具的输出）。
    返回 {correlated_risks:[{risk,dimensions}], multi_dim_flags, confidence}。
    confidence 为 0-1，多维度同向异常→高置信度；单维异常→低置信度（疑似误报）。"""
    import json
    try:
        ev = json.loads(evidence_json)
    except Exception:
        return {"error": "evidence_json 非合法 JSON"}
    if not isinstance(ev, dict):
        return {"error": "证据需为对象"}
    risks = []
    dims = []
    # 交叉1：节奏快(背答案) + 相似度高(抄写) = 抄袭/背答案
    tim = ev.get("timing", {})
    sim = ev.get("similarity", {})
    if tim.get("fast_questions") and sim.get("high_similarity"):
        risks.append({"risk": "背答案/抄袭(快答+高相似)",
                       "dimensions": ["timing", "similarity"]})
        dims += ["timing", "similarity"]
    # 交叉2：画面剧变(换人) + 节奏突变 = 中途换人代答
    cam = ev.get("camera", {})
    if cam.get("motion_rate", 0) > 0.4 and tim.get("rhythm_breaks"):
        risks.append({"risk": "中途换人代答(画面剧变+节奏突变)",
                       "dimensions": ["camera", "timing"]})
        dims += ["camera", "timing"]
    # 交叉3：突发高音(讨论) + 碎片式作答 = 朗读答案给他人
    aud = ev.get("audio", {})
    pat = ev.get("pattern", {})
    if aud.get("loud_rate", 0) > 0.1 and pat.get("fragment"):
        risks.append({"risk": "朗读答案(高音+碎片式作答)",
                       "dimensions": ["audio", "pattern"]})
        dims += ["audio", "pattern"]
    # 交叉4：无人(离席) + 批量补填 = 离开查资料后回来补填
    if cam.get("absence_rate", 0) > 0.3 and pat.get("batch_fill"):
        risks.append({"risk": "离席查资料后补填(无人+批量补填)",
                       "dimensions": ["camera", "pattern"]})
        dims += ["camera", "pattern"]
    # 置信度：涉及的独立维度数越多越高
    n_dims = len(set(dims))
    confidence = round(min(n_dims / 4, 1.0), 2)
    multi_flags = [r["risk"] for r in risks]
    return {"correlated_risks": risks, "multi_dim_flags": multi_flags,
            "n_dims_cross": n_dims, "confidence": confidence,
            "note": (f"多维度交叉验证发现{len(risks)}条强证据链，置信度{confidence}"
                     if risks else "无跨维度关联，单一维度异常疑似误报，置信度低")}


@tool
def predict_score_plausibility(sid: str, course: str, predicted_score: float) -> dict:
    """成绩合理性预测：对比该生历史答题正确率与本次得分，
    若本次远高于历史→疑似超常(作弊)；远低于→疑似摆烂/弃考。

    predicted_score 为本次考试的预估得分率(0-1)。
    返回 {history_acc, predicted, delta, plausibility, verdict}。
    plausibility: plausible(合理) / suspicious_high(异常高) / suspicious_low(异常低)。"""
    from .. import storage
    answers = storage.load_answers(sid)
    if not answers:
        return {"history_acc": 0, "predicted": predicted_score,
                "delta": 0, "plausibility": "no_baseline",
                "verdict": "无历史数据，无法判断合理性"}
    hist = sum(1 for a in answers if a.correct) / len(answers)
    delta = round(predicted_score - hist, 2)
    # 阈值：高出历史20%以上=异常高；低20%以上=异常低
    if delta > 0.2 and hist < 0.7:
        verdict = (f"本次得分率{predicted_score}高出历史{hist}共{delta}，"
                   "疑似超常发挥(需结合作答证据排查抄袭)")
        plaus = "suspicious_high"
    elif delta < -0.2:
        verdict = (f"本次得分率{predicted_score}低于历史{hist}共{delta}，"
                   "疑似弃考/摆烂(结合节奏判断是否中途放弃)")
        plaus = "suspicious_low"
    else:
        verdict = f"本次得分率与历史水平吻合(差{delta})，成绩合理"
        plaus = "plausible"
    return {"history_acc": round(hist, 2), "predicted": round(predicted_score, 2),
            "delta": delta, "plausibility": plaus, "verdict": verdict}


@tool
def scan_frame_vision(frame_log: str) -> dict:
    """摄像头关键帧视觉证据链：把前端逐帧视觉特征翻译成"画面发生了什么"的结构化证据。

    frame_log 为 JSON 数组，每帧 {"t":秒, "bright":0-255, "motion":0-1,
    "absent":bool, "motion_high":bool, "leave":bool}（前端每5秒抓帧的视觉特征）。
    工具对每帧做语义翻译（亮度→有人/无人/遮挡、运动量→稳定/剧变/多人嫌疑），
    标注异常时刻（无人、剧变、离席），串联成视觉证据链供 LLM 推理画面行为时序。
    返回 {n_frames, per_frame_desc, abnormal_moments, vision_summary}。"""
    import json
    try:
        log = json.loads(frame_log)
    except Exception:
        return {"error": "frame_log 非合法 JSON"}
    if not isinstance(log, list) or not log:
        return {"n_frames": 0, "msg": "无逐帧视觉数据（可能未授权摄像头）"}
    per_frame, abnorm = [], []
    for fr in log:
        if not isinstance(fr, dict):
            continue
        t = fr.get("t", 0)
        br = float(fr.get("bright", 0))
        mo = float(fr.get("motion", 0))
        absent = bool(fr.get("absent", False))
        mh = bool(fr.get("motion_high", False))
        leave = bool(fr.get("leave", False))
        # 语义翻译：把数值特征翻译成"画面描述"
        if absent or br < 18:
            scene = "画面近黑/无人(疑似离席或遮挡镜头)"
            kind = "absent"
        elif mh or mo > 0.25:
            scene = f"画面剧烈变化(motion={round(mo,2)},疑似多人/换人/动作剧烈)"
            kind = "motion_high"
        elif br < 40:
            scene = "画面偏暗(光线异常或低头)"
            kind = "dim"
        elif br > 200:
            scene = "画面过亮(疑似强光/屏幕反光)"
            kind = "bright"
        else:
            scene = "画面正常(有人正视屏幕)"
            kind = "normal"
        per_frame.append({"t": t, "scene": scene, "kind": kind})
        if kind != "normal":
            abnorm.append({"t": t, "kind": kind, "desc": scene})
    # 视觉证据链总结
    n_ab = len(abnorm)
    n_absent = sum(1 for a in abnorm if a["kind"] == "absent")
    n_motion = sum(1 for a in abnorm if a["kind"] == "motion_high")
    if n_absent >= 3:
        summ = f"全程{len(log)}帧中{n_absent}帧无人，疑似长时间离席"
    elif n_motion >= 2:
        summ = f"{n_motion}次画面剧变，疑似多人/中途换人"
    elif n_ab == 0:
        summ = f"全程{len(log)}帧视觉正常，未发现异常"
    else:
        summ = f"{n_ab}帧视觉异常，需结合答题节奏交叉验证"
    return {"n_frames": len(log), "per_frame_desc": per_frame[:12],
            "abnormal_moments": abnorm[:8],
            "n_abnormal": n_ab, "n_absent": n_absent, "n_motion_high": n_motion,
            "vision_summary": summ}


@tool
def patrol_timeline(events_json: str) -> dict:
    """AI 巡逻机器人：把整场考试的视觉/答题/音频多源事件按时间排序成统一巡逻时间线，
    自动发现跨源异常时序关联（如"离席→答题突变→高音"串联成疑似作弊链条）。

    events_json 为 JSON 数组，事件 {"t":秒, "source":"frame|answer|audio",
    "type":"absent|motion_high|fast|slow|break|loud|voice", "desc":"≤20字"}。
    工具按时间排序后扫描相邻异常事件的因果链，输出巡逻发现。
    返回 {timeline, patrol_findings, suspicious_sequences, patrol_verdict}。"""
    import json
    try:
        evs = json.loads(events_json)
    except Exception:
        return {"error": "events_json 非合法 JSON"}
    if not isinstance(evs, list) or not evs:
        return {"n": 0, "msg": "无巡逻事件"}
    evs = sorted([e for e in evs if isinstance(e, dict) and "t" in e], key=lambda x: x["t"])
    findings, seqs = [], []
    # 巡逻模式1：离席后短时间内答题突变 → 疑似离席查资料后回代答
    for i, e in enumerate(evs):
        if e.get("source") == "frame" and e.get("type") == "absent":
            for j in range(i + 1, min(i + 6, len(evs))):
                e2 = evs[j]
                if e2.get("t", 0) - e.get("t", 0) > 120:
                    break
                if e2.get("source") == "answer" and e2.get("type") in ("fast", "break", "slow"):
                    findings.append({"pattern": "离席后答题突变(疑似查资料后回代答)",
                                     "evidence": [e, e2], "severity": "high"})
                    seqs.append(f"t={e['t']}s离席→t={e2['t']}s{e2.get('desc','')}")
                    break
    # 巡逻模式2：画面剧变后音频高音 → 疑似多人讨论/换人交接
    for i, e in enumerate(evs):
        if e.get("source") == "frame" and e.get("type") == "motion_high":
            for j in range(i + 1, min(i + 4, len(evs))):
                e2 = evs[j]
                if e2.get("t", 0) - e.get("t", 0) > 60:
                    break
                if e2.get("source") == "audio" and e2.get("type") == "loud":
                    findings.append({"pattern": "画面剧变伴随高音(疑似多人讨论/交接)",
                                     "evidence": [e, e2], "severity": "high"})
                    seqs.append(f"t={e['t']}s画面剧变→t={e2['t']}s突发高音")
                    break
    # 巡逻模式3：连续多次答题突变（节奏断裂）→ 疑似中途换人
    ans_breaks = [e for e in evs if e.get("source") == "answer" and e.get("type") == "break"]
    if len(ans_breaks) >= 2:
        findings.append({"pattern": f"答题节奏{len(ans_breaks)}次突变(疑似中途换人)",
                         "evidence": ans_breaks[:3], "severity": "medium"})
        seqs.append(f"{len(ans_breaks)}次节奏断裂: " + "、".join(f"t={b['t']}s" for b in ans_breaks[:3]))
    # 巡逻裁决：发现高危链条→suspected；中危→watch；无→clear
    if any(f["severity"] == "high" for f in findings):
        verdict = "suspected_cheating"
    elif findings:
        verdict = "watch"
    else:
        verdict = "clear"
    return {"timeline": [{"t": e.get("t"), "source": e.get("source"),
                          "type": e.get("type"), "desc": e.get("desc", "")} for e in evs[:20]],
            "patrol_findings": findings[:5],
            "suspicious_sequences": seqs[:5],
            "n_findings": len(findings),
            "patrol_verdict": verdict,
            "note": (f"AI巡逻机器人扫描{len(evs)}个事件，发现{len(findings)}条异常时序关联"
                     if findings else f"AI巡逻机器人扫描{len(evs)}个事件，未发现跨源异常时序关联")}


@tool
def issue_verdict(sid: str, course: str, verdict: str) -> dict:
    """出具监考裁决书并落盘。verdict 为 JSON 字符串，形如：
    {"risk_level":"low|medium|high","flags":["节奏异常","内容抄袭"],
     "evidence":"证据摘要(≤40字)","action":"处置建议(≤40字)",
     "score_adjustment":"是否调整成绩的说明"}。"""
    import json, os
    from .. import storage
    try:
        v = json.loads(verdict)
        if not isinstance(v, dict) or "risk_level" not in v:
            raise ValueError
    except Exception:
        return {"error": "verdict 必须含 risk_level 的 JSON"}
    rec = {"sid": sid, "course": course, "verdict": v,
           "ts": storage.now_iso()}
    p = os.path.join(storage.DATA_DIR, f"proctor_{sid}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ok": True, "risk_level": v["risk_level"],
            "flags": v.get("flags", []), "action": v.get("action", "")}


PROCTOR_SYS = (
    "你是博士级在线考试 AI 防作弊监考官，像真实考场监考长一样全程『看着』学生作答："
    "基于摄像头图像视觉证据链 + AI 巡逻机器人时间线 + 多维交叉验证，做实时高水平取证裁决。"
    "可调用工具：scan_timing（查答题节奏：过快/过慢/突变/前后段差异）、"
    "scan_similarity（查作答与参考答案相似度：字面重合+关键词覆盖+疑似抄写题号）、"
    "scan_pattern（查行为模式：突发式/碎片式/模板式/批量补填）、"
    "scan_camera（摄像头视觉统计取证：无人率/离开次数/画面剧变/拒绝授权）、"
    "scan_frame_vision（摄像头逐帧视觉证据链：把每帧画面特征翻译成『画面发生了什么』，"
    "标注无人/剧变/离席时刻，供推理画面行为时序——这是『有图像』的核心取证）、"
    "scan_audio（麦克风语音取证：无声率/突发高音/多人喧哗嫌疑/未授权）、"
    "cross_validate（多维证据交叉验证：找多维度同向异常强证据链，降误报）、"
    "patrol_timeline（AI 巡逻机器人：把视觉/答题/音频多源事件按时间排序，"
    "自动发现跨源异常时序关联如『离席→答题突变→高音』疑似作弊链条——这是『AI巡逻』核心）、"
    "predict_score_plausibility（成绩合理性预测：对比历史正确率，异常高/低）、"
    "issue_verdict（出具裁决书，唯一生效入口）。"
    "流程：必须先 scan_timing 读节奏，再 scan_similarity 查相似度，"
    "必须 scan_camera 查摄像头视觉统计，"
    "若有逐帧视觉数据则必须 scan_frame_vision 看『画面发生了什么』（逐帧视觉证据链），"
    "必须 scan_audio 查音频证据，"
    "若任一异常则 scan_pattern 深挖行为模式，"
    "然后必须 cross_validate 把五维证据做交叉验证找出强证据链，"
    "必须 patrol_timeline 让 AI 巡逻机器人扫描全部事件时间线找跨源异常时序关联，"
    "必须 predict_score_plausibility 对比本次得分与历史正确率判断成绩合理性，"
    "最后必须调用 issue_verdict 工具出具裁决书（risk_level/flags/evidence/action）。"
    "裁决原则：证据说话、宁可漏抓不可错判、对正常考生不得误伤；"
    "单一维度异常置信度低不得定作弊，须多维度交叉验证同向 + AI巡逻时序关联方可升级风险。"
    "issue_verdict 調用成功后，再用≤110字向监考长汇报：风险等级、关键证据"
    "（含逐帧视觉/AI巡逻发现/交叉验证强证据链）、处置建议。"
)


PROCTOR_TOOLS = [scan_timing, scan_similarity, scan_pattern, scan_camera,
                 scan_frame_vision, scan_audio, cross_validate,
                 patrol_timeline, predict_score_plausibility, issue_verdict]


def run_proctor(sid: str, course: str, submit_log: list[dict],
                camera: dict | None = None,
                audio: dict | None = None) -> dict:
    """AI 监考官 ReAct：对作答日志+摄像头逐帧视觉证据链+麦克风音频+AI巡逻时间线
    全景取证 → 交叉验证 → 裁决 → 返回 {steps, answer, verdict, patrol}。

    submit_log: [{"i":0,"dur":120,"ans":"...","ref":"..."}, ...]
    camera: 前端摄像头取证 {"authorized":true,"shots":30,...,
            "frame_log":[{t,bright,motion,absent,motion_high,leave},...],  # 逐帧视觉特征
            "events":[{t,source,type,desc},...]}  # 巡逻事件（可选）
    audio:  前端麦克风音量统计 {"authorized":true,"samples":30,...}（可选）
    """
    import json, os, time
    from .. import storage
    model = RouterChatModel(temperature=0.3, max_tokens=650)
    log_str = json.dumps(submit_log, ensure_ascii=False)
    cam_meta = {k: v for k, v in (camera or {}).items()
                if k not in ("frames", "frame_log", "events")}
    cam_str = json.dumps(cam_meta, ensure_ascii=False)
    aud_str = json.dumps(audio or {}, ensure_ascii=False)
    frame_log = (camera or {}).get("frame_log") or []
    events = (camera or {}).get("events") or []
    fl_str = json.dumps(frame_log, ensure_ascii=False)
    ev_str = json.dumps(events, ensure_ascii=False)
    msg = (f"学生{sid}刚完成《{course}》在线考试。作答日志、摄像头视觉统计、逐帧视觉特征、"
           f"麦克风音频统计、AI巡逻事件如下，请像真实监考长一样全程取证并出具裁决书。\n"
           f"submit_log={log_str}\n"
           f"camera={cam_str}\n"
           f"frame_log(逐帧视觉特征,供scan_frame_vision)={fl_str}\n"
           f"audio={aud_str}\n"
           f"events(巡逻事件,供patrol_timeline)={ev_str}")
    t0 = time.time()
    try:
        steps, answer = _run_react(model, PROCTOR_TOOLS, PROCTOR_SYS, msg)
    except Exception as e:
        return {"steps": [], "answer": f"监考系统故障：{e}",
                "verdict": {"risk_level": "unknown"}}
    # 保底1：ReAct 若漏调扫描工具（摄像头/音频证据未走工具链），强制补调并记入轨迹
    if camera and not any(s.get("tool") == "scan_camera" for s in steps):
        try:
            res = scan_camera.invoke({"camera_json": cam_str})
            steps.append({"tool": "scan_camera",
                          "args": {"camera_json": cam_str[:60] + "..."},
                          "result": json.dumps(res, ensure_ascii=False)[:400]})
        except Exception:
            pass
    # 保底1a：逐帧视觉证据链——前端采集的 frame_log 必须走 scan_frame_vision
    if frame_log and not any(s.get("tool") == "scan_frame_vision" for s in steps):
        try:
            res = scan_frame_vision.invoke({"frame_log": fl_str})
            steps.append({"tool": "scan_frame_vision",
                          "args": {"frame_log": fl_str[:60] + "..."},
                          "result": json.dumps(res, ensure_ascii=False)[:400]})
        except Exception:
            pass
    if audio and not any(s.get("tool") == "scan_audio" for s in steps):
        try:
            res = scan_audio.invoke({"audio_json": aud_str})
            steps.append({"tool": "scan_audio",
                          "args": {"audio_json": aud_str[:60] + "..."},
                          "result": json.dumps(res, ensure_ascii=False)[:400]})
        except Exception:
            pass
    # 保底1b：强制做交叉验证——把已采集的五维证据合并喂给 cross_validate
    # 即使 ReAct 调过，也用合并后的完整证据再跑一次，确保强证据链不漏
    if not any(s.get("tool") == "cross_validate" for s in steps):
        ev_map = _gather_evidence(steps)
        try:
            res = cross_validate.invoke({"evidence_json": json.dumps(ev_map, ensure_ascii=False)})
            steps.append({"tool": "cross_validate",
                          "args": {"evidence_json": "..."},
                          "result": json.dumps(res, ensure_ascii=False)[:400]})
        except Exception:
            pass
    # 保底1c：AI 巡逻机器人——多源事件时间线必须走 patrol_timeline 找跨源异常时序关联
    patrol_res = None
    if events and not any(s.get("tool") == "patrol_timeline" for s in steps):
        try:
            patrol_res = patrol_timeline.invoke({"events_json": ev_str})
            steps.append({"tool": "patrol_timeline",
                          "args": {"events_json": ev_str[:60] + "..."},
                          "result": json.dumps(patrol_res, ensure_ascii=False)[:400]})
        except Exception:
            pass
    elif any(s.get("tool") == "patrol_timeline" for s in steps):
        for s in steps:
            if s.get("tool") == "patrol_timeline":
                try:
                    patrol_res = json.loads(s.get("result", "{}"))
                except Exception:
                    patrol_res = None
                break
    # 保底1d：成绩合理性预测——从作答日志估本次得分率，对比历史
    if not any(s.get("tool") == "predict_score_plausibility" for s in steps):
        try:
            pred = _estimate_score(submit_log)
            res = predict_score_plausibility.invoke({"sid": sid, "course": course,
                                                     "predicted_score": pred})
            steps.append({"tool": "predict_score_plausibility",
                          "args": {"sid": sid, "course": course, "predicted_score": pred},
                          "result": json.dumps(res, ensure_ascii=False)[:400]})
        except Exception:
            pass
    # 保底2：若 ReAct 没有调用 issue_verdict 落盘（文件未在本次运行中更新），
    # 则根据已采集证据做确定性裁决并强制落盘——保证裁决书永远与本次证据一致
    p = os.path.join(storage.DATA_DIR, f"proctor_{sid}.json")
    verdict, verdict_fresh = {}, False
    if os.path.exists(p):
        try:
            verdict = json.load(open(p, encoding="utf-8")).get("verdict", {})
            verdict_fresh = os.path.getmtime(p) >= t0
        except Exception:
            pass
    has_issue = any(s.get("tool") == "issue_verdict" for s in steps)
    if not has_issue or not verdict_fresh:
        v = _deterministic_verdict(steps, sid, course)
        try:
            issue_verdict.invoke({"sid": sid, "course": course,
                                  "verdict": json.dumps(v, ensure_ascii=False)})
            verdict_fresh = True
        except Exception:
            pass
        verdict = v
        if not has_issue:
            steps.append({"tool": "issue_verdict",
                          "args": {"sid": sid, "course": course,
                                   "verdict": json.dumps(v, ensure_ascii=False)[:60] + "..."},
                          "result": f"落盘保底裁决 ok risk_level={v['risk_level']}"})
    return {"steps": steps, "answer": answer, "verdict": verdict,
            "patrol": patrol_res or {"patrol_verdict": "no_events"}}


def _deterministic_verdict(steps: list, sid: str, course: str) -> dict:
    """从已采集的工具证据做确定性裁决（LLM 漏调 issue_verdict 时的保底）。
    工具结果可能被 _run_react 截断到 180 字符，故 json.loads 失败时用正则宽容提取。"""
    import json, re
    flags: list[str] = []
    high, medium = 0, 0
    for s in steps:
        raw = s.get("result") or ""
        tool = s.get("tool", "")
        try:
            r = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(r, dict):
                r = {}
        except Exception:
            r = {}
        if not r and isinstance(raw, str) and raw.startswith("{"):
            # 截断 JSON 的宽容提取：flags 数组 + 关键数值
            m = re.search(r'"(\w*flags)":\s*\[(.*?)\]', raw)
            if m:
                r = {"flags_items": re.findall(r'"([^"]+)"', m.group(2))}
            for k, var in (("absence_rate", "abs"), ("motion_rate", "mot"), ("loud_rate", "loud")):
                mm = re.search(rf'"{k}":\s*([0-9.]+)', raw)
                if mm:
                    r[var] = float(mm.group(1))
        if s.get("tool") == "scan_camera":
            flags += r.get("camera_flags", []) or r.get("flags_items", [])
            if r.get("authorized") is False or "拒绝摄像头授权" in raw:
                medium += 1   # 拒绝监考取证本身即风险信号
            elif r.get("mot", r.get("motion_rate", 0)) > 0.4 or r.get("abs", r.get("absence_rate", 0)) > 0.5:
                high += 1
            elif r.get("abs", r.get("absence_rate", 0)) > 0.3 or "离席" in raw or "多次离开" in raw:
                medium += 1
        elif s.get("tool") == "scan_audio":
            flags += r.get("audio_flags", []) or r.get("flags_items", [])
            if r.get("authorized") is False or "麦克风未授权" in raw:
                medium += 1   # 拒绝音频取证本身即风险信号
            elif r.get("loud", r.get("loud_rate", 0)) > 0.25 or "喧哗" in raw:
                medium += 1
        elif s.get("tool") == "scan_similarity":
            hs = r.get("high_similarity")
            if hs is None:
                low_raw = raw.lower()
                hs = ('"high_similarity": true' in low_raw
                      or "'high_similarity': true" in low_raw)
            if hs:
                flags.append("作答与参考答案高度相似(抄袭嫌疑)")
                medium += 1
        elif s.get("tool") == "scan_pattern":
            pat = r.get("pattern", "")
            if not pat:
                mm = re.search(r'"pattern":\s*"([^"]+)"', raw)
                pat = mm.group(1) if mm else ""
            if "疑似" in pat:
                flags.append(pat)
                medium += 1
        elif s.get("tool") == "cross_validate":
            # 交叉验证：多维度同向异常→置信度高的强证据链，直接升级风险
            conf = r.get("confidence", 0)
            risks_list = r.get("multi_dim_flags", []) or r.get("correlated_risks", [])
            if isinstance(risks_list, list) and risks_list:
                flags += risks_list[:4]
                if conf >= 0.75:
                    high += 1
                elif conf >= 0.5:
                    medium += 1
        elif s.get("tool") == "scan_frame_vision":
            # 逐帧视觉证据链：无人帧/剧变帧多则升级风险
            n_ab = r.get("n_abnormal", 0)
            n_abs = r.get("n_absent", 0)
            n_mot = r.get("n_motion_high", 0)
            summ = r.get("vision_summary", "") or raw
            if n_abs >= 3:
                flags.append("逐帧视觉:长时间无人(疑似离席)")
                high += 1
            elif n_mot >= 2:
                flags.append("逐帧视觉:多次画面剧变(疑似多人/换人)")
                high += 1
            elif n_ab > 0:
                flags.append(f"逐帧视觉:{n_ab}帧异常")
                medium += 1
            if "疑似" in summ and "长时间离席" not in " ".join(flags):
                flags.append("逐帧视觉异常需交叉验证")
        elif s.get("tool") == "patrol_timeline":
            # AI 巡逻机器人：跨源异常时序关联是强证据，直接升级风险
            pv = r.get("patrol_verdict", "")
            pfinds = r.get("patrol_findings", []) or []
            seqs = r.get("suspicious_sequences", []) or []
            if isinstance(pfinds, list) and pfinds:
                flags += [f.get("pattern", "") for f in pfinds[:3] if f.get("pattern")]
            if isinstance(seqs, list) and seqs:
                flags += seqs[:2]
            if pv == "suspected_cheating":
                high += 1   # 巡逻发现高危链条
            elif pv == "watch":
                medium += 1
        elif s.get("tool") == "predict_score_plausibility":
            plaus = r.get("plausibility", "")
            if plaus == "suspicious_high":
                flags.append("成绩异常高于历史(疑似超常)")
                medium += 1
            elif plaus == "suspicious_low":
                flags.append("成绩异常低于历史(疑似弃考)")
    risk = "high" if high >= 2 or (high >= 1 and medium >= 2) else "medium" if medium else "low"
    if not flags:
        flags = ["无异常"]
    return {"risk_level": risk, "flags": flags[:8],
            "evidence": f"作答节奏/相似度/摄像头/音频/交叉验证/成绩预测六维取证共{len(steps)}项",
            "action": ("立即人工复核" if risk == "high"
                       else "建议抽查复核" if risk == "medium" else "正常归档"),
            "score_adjustment": "由人工复核决定"}


def _gather_evidence(steps: list) -> dict:
    """从已采集的 scan_* 工具结果中提取五维证据，合并成 cross_validate 的输入。"""
    import json
    ev: dict = {}
    key_map = {"scan_timing": "timing", "scan_similarity": "similarity",
               "scan_pattern": "pattern", "scan_camera": "camera", "scan_audio": "audio"}
    for s in steps:
        tool = s.get("tool", "")
        if tool not in key_map:
            continue
        raw = s.get("result") or ""
        try:
            r = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(r, dict):
                r = {}
        except Exception:
            r = {}
        ev[key_map[tool]] = r
    return ev


def _estimate_score(submit_log: list[dict]) -> float:
    """从作答日志粗估本次得分率：有 ref 的题按字面相似度估，无 ref 取 0.5。"""
    import re
    if not submit_log:
        return 0.5
    scores = []
    for x in submit_log:
        ans = (x.get("ans", "") or "").strip()
        ref = (x.get("ref", "") or "").strip()
        if not ans:
            scores.append(0.0)
        elif not ref:
            scores.append(0.5)
        else:
            a_set = set(re.sub(r"[，。、,.！!？?；;：:\s]+", "", ans))
            r_set = set(re.sub(r"[，。、,.！!？?；;：:\s]+", "", ref))
            scores.append(len(a_set & r_set) / max(len(a_set | r_set), 1))
    return round(sum(scores) / len(scores), 2) if scores else 0.5


# ============================================================
# 考中实时 AI 巡逻机器人（轻量 ReAct，前端检测到异常帧时调用）
# ============================================================
PATROL_SYS = (
    "你是考中实时 AI 巡逻机器人，像真实监考员在考场巡视一样，"
    "对前端刚上报的异常帧+近期事件做快速研判，决定是否需要语音警告考生并给出警告话术。"
    "可调用工具：scan_frame_vision（看逐帧画面发生了什么）、"
    "patrol_timeline（扫描近期事件时间线找跨源异常时序关联）。"
    "流程：先 scan_frame_vision 看当前帧及近期帧画面，再 patrol_timeline 看近期事件时序关联。"
    "输出严格 JSON：{\"warn\":true/false, \"text\":\"≤40字中文警告话术\", "
    "\"level\":\"normal|watch|suspected\"}。正常画面或单一偶发异常→warn=false；"
    "发现疑似离席/多人/换人/朗读答案等强证据→warn=true 并给出具体话术（指出检测到什么行为）。"
)


def run_patrol(sid: str, course: str, frame_log: list[dict],
               events: list[dict]) -> dict:
    """考中实时 AI 巡逻：前端检测到异常帧时上报当前帧+近期事件，
    本函数用轻量 ReAct 快速研判是否需语音警告考生，返回 {warn, text, level, patrol_verdict}。

    frame_log: 当前及近期帧的视觉特征 [{t,bright,motion,absent,motion_high,leave},...]
    events: 近期巡逻事件 [{t,source,type,desc},...]
    """
    import json
    if not frame_log and not events:
        return {"warn": False, "text": "无异常", "level": "normal",
                "patrol_verdict": "clear"}
    # 先跑两个工具拿确定性证据（避免 LLM 多轮延迟，考中巡逻要快）
    fl_res = scan_frame_vision.invoke({"frame_log": json.dumps(frame_log, ensure_ascii=False)}) \
        if frame_log else {"n_frames": 0}
    pt_res = patrol_timeline.invoke({"events_json": json.dumps(events, ensure_ascii=False)}) \
        if events else {"patrol_verdict": "no_events", "patrol_findings": []}
    # 确定性研判：工具已给出强证据，直接据此裁决（快、稳）
    pv = pt_res.get("patrol_verdict", "clear")
    n_abs = fl_res.get("n_absent", 0)
    n_mot = fl_res.get("n_motion_high", 0)
    findings = pt_res.get("patrol_findings", [])
    seqs = pt_res.get("suspicious_sequences", [])
    # 拼警告话术证据摘要喂 LLM 让它用人话表述
    evidence_brief = (f"逐帧视觉:{fl_res.get('vision_summary','无')}; "
                      f"巡逻发现:{len(findings)}条, 时序:{'; '.join(seqs[:2]) or '无'}; "
                      f"巡逻裁决:{pv}")
    # 确定性阈值：高危链条或长时间无人→必须警告
    must_warn = (pv == "suspected_cheating" or n_abs >= 3 or n_mot >= 2)
    if must_warn:
        level = "suspected" if pv == "suspected_cheating" else "watch"
    elif pv == "watch" or findings:
        level = "watch"
    else:
        level = "normal"
    # LLM 生成自然中文警告话术（单次调用，不走 ReAct 多轮，保证实时性）
    text = ""
    try:
        from ..llm import get_llm
        sys2 = ("你是考中监考员。根据以下实时取证证据，用≤40字中文给出一段"
                "对考生的语音警告话术（直接指出检测到什么行为、要求怎么做）。"
                "若无强证据则输出空字符串。只输出话术本身，不要解释。")
        text = get_llm().chat(sys2, evidence_brief, temperature=0.4, max_tokens=80).strip()
    except Exception:
        text = ""
    warn = bool(must_warn or (level == "watch" and (n_abs or n_mot or findings)))
    if not text and warn:
        # LLM 失败兜底话术
        if n_abs >= 3:
            text = "监考提醒：检测到您长时间离开画面，请立即回到座位保持人在镜头内。"
        elif n_mot >= 2:
            text = "监考提醒：画面多次剧烈变化，请勿让他人出现在镜头前。"
        elif findings:
            text = "监考提醒：检测到异常行为时序，请保持独立作答。"
    return {"warn": warn, "text": text, "level": level,
            "patrol_verdict": pv, "evidence": evidence_brief,
            "n_abnormal_frames": fl_res.get("n_abnormal", 0),
            "n_findings": len(findings)}
