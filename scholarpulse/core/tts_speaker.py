"""离线角色配音：Windows SAPI5 逐行合成台词 → 按视频时间线混合为单条 WAV。

设计要点：
- 全离线（不联网、不依赖云 API），SAPI5 引擎为系统自带
- 每个角色 key 经哈希 → 固定语速/声线（同一角色全剧一致，像同一演员配音）
- 单行超过 max_ms 自动提高语速重合成一次，仍超长则按 max_ms+400ms 截断
- numpy float32 加法混合 → 16bit mono WAV（22050Hz），前端 decodeAudioData 自动重采样

修复记录：pyttsx3 同一引擎第二次 save_to_file+runAndWait 会永久死锁（SAPI5 循环
线程僵死，实测第 1 句成功、第 2 句卡死）。生产路径改为每行直连 SAPI5 COM 同步
Speak（显式传入 engine 仍走原 pyttsx3 接口，供测试 mock）；直连失败时回退
"每行全新 pyttsx3 引擎"——绝不跨行复用引擎。
"""
import hashlib
import os
import shutil
import tempfile
import time
import wave

import numpy as np

SR = 22050


def _init_engine():
    try:
        import pyttsx3
        return pyttsx3.init()
    except Exception:
        return None


_VOICE_IDS_CACHE = None


def _sapi_voice_ids_cached() -> list:
    """直连 SAPI5 枚举语音 token id（与 pyttsx3 同源注册表 id，进程内缓存一次）。

    必须与 _pick_voices 同规则做"中文声优先"过滤：否则角色哈希可能命中英文声
    （如 Zira），中文台词合成出来是 46 字节空 WAV。
    """
    global _VOICE_IDS_CACHE
    if _VOICE_IDS_CACHE is None:
        try:
            import pythoncom
            import win32com.client
            if not getattr(_COM, "init", False):
                pythoncom.CoInitialize()
                _COM.init = True
            v = win32com.client.Dispatch("SAPI.SpVoice")
            toks = v.GetVoices()
            all_ids = [toks.Item(i).Id for i in range(toks.Count)]
            keys = ("zh-", "zh/", "chinese", "huihui", "kangkang", "yaoyao")
            zh = [i for i in all_ids if any(k in i.lower() for k in keys)]
            _VOICE_IDS_CACHE = zh or all_ids or [""]
            v = toks = None
        except Exception:
            _VOICE_IDS_CACHE = [""]
    return _VOICE_IDS_CACHE


_COM = __import__("threading").local()


def _sapi_synth(text: str, voice_id: str, rate: int, out_path: str) -> bool:
    """直连 SAPI5 COM 同步合成（v.Speak 说完才返回，无后台循环线程，不产生死锁）。

    COM 生命周期：每线程只 CoInitialize 一次且不 Uninit（工作线程复用场景下
    反复 init/uninit 会破坏 COM 状态——实测第 2/4 次调用产出 46 字节空 WAV）；
    COM 对象必须在退出前显式释放，严禁留到 CoUninitialize 之后被 GC。
    """
    try:
        import pythoncom
        import win32com.client
        if not getattr(_COM, "init", False):
            pythoncom.CoInitialize()
            _COM.init = True
        v = stream = toks = None
        try:
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(out_path, 3, False)                     # SSFMCreateForWrite
            v = win32com.client.Dispatch("SAPI.SpVoice")
            toks = v.GetVoices()
            for i in range(toks.Count):
                if toks.Item(i).Id == voice_id:
                    v.Voice = toks.Item(i)
                    break
            v.Rate = max(-10, min(10, int((rate - 196) / 8)))   # 172-219 语速域 → SAPI -3..+3
            v.AudioOutputStream = stream
            v.Speak(text)
            stream.Close()                                      # 落盘并写完整 RIFF 头
            ok = os.path.exists(out_path) and os.path.getsize(out_path) > 44
            return ok
        except Exception:
            return False
        finally:
            v = stream = toks = None                            # 先释放 COM 对象再返回
    except Exception:
        return False


def _pick_voices(eng) -> list:
    """优先中文语音，退回全部语音，最后给空串兜底。"""
    vs = eng.getProperty("voices") or []
    keys = ("zh-", "zh/", "chinese", "huihui", "kangkang", "yaoyao")
    zh = [v.id for v in vs
          if any(k in f"{getattr(v, 'id', '')} {getattr(v, 'name', '')}".lower() for k in keys)]
    return zh or [v.id for v in vs] or [""]


def _style_for(key: str, voice_ids: list) -> tuple:
    """角色 key → (voice_id, rate)。同一 key 全剧稳定。"""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return voice_ids[h % len(voice_ids)], 172 + (h >> 8) % 48   # 172-219 语速微差


def _synth_one(eng, text: str, voice_id: str, rate: int, out_path: str,
               timeout_s: float = 25.0) -> bool:
    """合成单行到 out_path。pyttsx3 落盘是异步的，需轮询文件稳定。"""
    try:
        eng.setProperty("voice", voice_id)
        eng.setProperty("rate", rate)
        eng.save_to_file(text, out_path)
        eng.runAndWait()
        size, stable, deadline = -1, 0, time.time() + timeout_s
        while time.time() < deadline:
            s = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            if s >= 44 and s == size:
                stable += 1
                if stable >= 2:
                    return True
            else:
                stable = 0
            size = s
            time.sleep(0.05)
        return False
    except Exception:
        return False


def _read_wav(path: str):
    """读 wav 为 float32 mono（SR 归一化）。失败返回 None。"""
    try:
        with wave.open(path, "rb") as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        if sw != 2 or not n:
            return None
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch)[:, 0]
        if sr != SR and len(x):
            n_out = int(len(x) * SR / sr)
            t = np.arange(n_out, dtype=np.float32) * (sr / SR)
            x = np.interp(t, np.arange(len(x), dtype=np.float32), x).astype(np.float32)
        return x
    except Exception:
        return None


def _to_wav(mix: np.ndarray) -> bytes:
    """float32 [-1,1] → 16bit mono WAV bytes。"""
    pcm = (np.clip(mix, -1.0, 1.0) * 32767).astype("<i2")
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def render_speech(rows: list, total_ms: int, engine=None) -> bytes:
    """按时间线合成并混合配音。rows: [{text, key, offset_ms, max_ms}]。

    返回整片单声道 WAV bytes（长度 = total_ms + 0.5s 余量）；TTS 不可用时返回静音轨，
    由前端照常出片（只有 BGM 与字幕），不因 TTS 失败而失败。

    生产路径（engine=None）：环境探测走直连枚举（绝不创建 pyttsx3 引擎——其 SAPI
    循环线程即使"弃用"仍存活，会与直连 COM 竞态产生空 WAV）；仅当直连不可用时才
    尝试 pyttsx3（每行全新引擎，绝不复用）。显式传入 engine（测试 mock）走原接口。
    """
    total_n = SR * max(1, int(total_ms)) // 1000
    mix = np.zeros(total_n + SR // 2, dtype=np.float32)
    if not rows:
        return _to_wav(mix)
    voice_ids = None
    if engine is not None:
        voice_ids = _pick_voices(engine)                # 测试/显式引擎
    else:
        voice_ids = _sapi_voice_ids_cached()
        if voice_ids == [""]:                           # 直连不可用 → pyttsx3-only 环境
            probe = _init_engine()
            if probe is None:                           # 完全无 TTS → 静音兜底
                return _to_wav(mix)
            del probe                                   # 探测引擎即弃（此分支不会与直连并存）

    def _synth_line(text: str, key: str, vid: str, rt: int, p: str) -> bool:
        if engine is not None:                          # 测试/显式引擎：原 pyttsx3 接口
            return _synth_one(engine, text, vid, rt, p)
        if voice_ids and _sapi_synth(text, vid, rt, p):  # 生产主路径：直连 COM，同步无死锁
            return True
        eng_row = _init_engine()                        # 回退：每行全新引擎，绝不跨行复用
        if eng_row is None:
            return False
        vid2, rt2 = _style_for(key, _pick_voices(eng_row))
        return _synth_one(eng_row, text, vid2, rt2, p)

    tmp = tempfile.mkdtemp(prefix="sp_tts_")
    try:
        for i, r in enumerate(rows):
            text = str(r.get("text", "")).strip()[:60]
            if not text:
                continue
            off = int(r.get("offset_ms", 0))
            max_ms = max(800, int(r.get("max_ms", 6000)))
            key = str(r.get("key") or "旁白")
            vid, rate = _style_for(key, voice_ids or [""])
            p = os.path.join(tmp, f"{i}.wav")
            if not _synth_line(text, key, vid, rate, p):
                continue
            pcm = _read_wav(p)
            if pcm is None or not len(pcm):
                continue
            dur_ms = len(pcm) / SR * 1000
            if dur_ms > max_ms:                                   # 超长 → 提速重合成一次
                rate2 = min(300, int(rate * dur_ms / max_ms) + 10)
                if _synth_line(text, key, vid, rate2, p):
                    pcm2 = _read_wav(p)
                    if pcm2 is not None and len(pcm2):
                        pcm = pcm2
            cap = max(1, min(len(pcm), SR * (max_ms + 400) // 1000))  # 最多溢出行尾 0.4s
            s0 = SR * off // 1000
            if s0 >= len(mix):
                continue
            seg = pcm[:cap].astype(np.float32) * 0.95
            e = min(len(mix), s0 + len(seg))
            mix[s0:e] += seg[:e - s0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _to_wav(mix)
