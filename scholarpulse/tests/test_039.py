"""离线角色配音测试（mock SAPI5 引擎）：声线稳定/超长提速/偏移混合/WAV 格式。"""
import io
import wave

import numpy as np

from scholarpulse.core import tts_speaker as TS
from scholarpulse.core.tts_speaker import SR, _pick_voices, _style_for, _read_wav, render_speech


class FakeVoice:
    def __init__(self, vid):
        self.id, self.name = vid, vid


class FakeEngine:
    """save_to_file + runAndWait 后写出固定时长的正弦 wav；记录每次 rate。"""

    def __init__(self, dur_s=1.0, write=True):
        self.dur, self.write, self.rates = dur_s, write, []
        self.props, self.saved = {}, []

    def getProperty(self, k):
        if k == "voices":
            return [FakeVoice("zh-a"), FakeVoice("en-b")]
        return self.props.get(k)

    def setProperty(self, k, v):
        self.props[k] = v
        if k == "rate":
            self.rates.append(v)

    def save_to_file(self, text, path):
        self.saved.append(path)

    def runAndWait(self):
        path = self.saved[-1]
        if not self.write:
            return
        n = int(SR * self.dur)
        pcm = (np.sin(np.linspace(0, 400, n)) * 8000).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())


def _decode(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def test_voice_pick_prefers_chinese():
    eng = FakeEngine()
    assert _pick_voices(eng) == ["zh-a"]                     # 中文声优先


def test_style_for_stable_and_spread():
    vs = ["zh-a", "en-b"]
    a1 = _style_for("小七", vs)
    assert a1 == _style_for("小七", vs)                       # 同角色全剧稳定
    styles = {_style_for(k, vs)[1] for k in ["小七", "阿泽", "林老师", "旁白"]}
    assert len(styles) >= 2                                  # 语速有区分
    assert 172 <= min(styles) and max(styles) <= 219


def test_render_speech_offsets_and_format():
    eng = FakeEngine(dur_s=1.0)
    rows = [{"text": "你好世界", "key": "小七", "offset_ms": 0, "max_ms": 3000},
            {"text": "第二句台词", "key": "阿泽", "offset_ms": 2000, "max_ms": 3000}]
    pcm = _decode(render_speech(rows, 5000, engine=eng))
    assert len(pcm) == SR * 5 + SR // 2                      # total + 0.5s 余量
    assert np.abs(pcm[: SR]).max() > 0.05                    # 第 0-1s 有声
    assert np.abs(pcm[int(SR * 1.2): int(SR * 1.9)]).max() < 0.01   # 间隙静音
    assert np.abs(pcm[int(SR * 2.1): int(SR * 3)]).max() > 0.05     # 第 2-3s 有声
    assert np.abs(pcm).max() <= 1.0                          # 混合不削波


def test_overlong_row_speeds_up_then_caps():
    eng = FakeEngine(dur_s=3.0)                              # 引擎固定输出 3 秒
    rows = [{"text": "这一句很长很长很长", "key": "小七", "offset_ms": 0, "max_ms": 1000}]
    pcm = _decode(render_speech(rows, 4000, engine=eng))
    assert eng.rates[1] > eng.rates[0]                       # 第二次合成提速了
    assert len(eng.rates) == 2                               # 只重试一次
    # 截断到 max_ms + 400ms
    seg = pcm[: int(SR * 1.6)]
    assert np.abs(seg[int(SR * 1.45):]).max() < 0.01


def test_failed_rows_skipped_silence():
    eng = FakeEngine(write=False)                            # 引擎不落盘 → 行失败
    rows = [{"text": "hello", "key": "小七", "offset_ms": 0, "max_ms": 2000}]
    pcm = _decode(render_speech(rows, 3000, engine=eng))
    assert np.abs(pcm).max() == 0.0                          # 静音轨但不崩溃


def test_empty_rows_and_engine_none(monkeypatch):
    pcm = _decode(render_speech([], 2000, engine=None))
    assert len(pcm) == SR * 2 + SR // 2 and np.abs(pcm).max() == 0.0
    # 模拟无 TTS 环境：直连枚举与 pyttsx3 均不可用
    monkeypatch.setattr(TS, "_sapi_voice_ids_cached", lambda: [""])
    monkeypatch.setattr(TS, "_init_engine", lambda: None)
    pcm2 = _decode(render_speech([{"text": "x", "key": "k", "offset_ms": 0, "max_ms": 1000}],
                                 2000, engine=None))
    assert np.abs(pcm2).max() == 0.0                         # 无引擎 → 静音兜底


def test_offset_beyond_total_no_crash():
    eng = FakeEngine(dur_s=0.5)
    rows = [{"text": "越界台词", "key": "小七", "offset_ms": 99999, "max_ms": 1000}]
    pcm = _decode(render_speech(rows, 2000, engine=eng))
    assert np.abs(pcm).max() == 0.0


def test_read_wav_resamples():
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "r.wav")
    n = 1100                                                  # 11025Hz 0.1s
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(11025)
        w.writeframes((np.ones(n) * 12000).astype("<i2").tobytes())
    x = _read_wav(p)
    assert len(x) == 2200 == int(1100 * SR / 11025)           # 重采样到 22050
    assert abs(float(x.mean()) - 12000 / 32768) < 0.01
