"""创意内容生成工坊测试（全离线）：brief 注入/失败返错不套模板/场景切分/TTS 净化/多格式导出。"""
import re

from scholarpulse.core import creative_studio as CS

OUTLINE_DEMO = "第一幕·起：平静被打破\n第二幕·承：问题升级\n第三幕·转：意外反转\n第四幕·合：收束升华"


def test_speak_text_cleans_and_limits():
    raw = "甲：你好【舞台】(侧幕) #乱*符`号\n乙：第二句很长" + "呀" * 400
    t = CS.speak_text(raw, limit=50)
    assert "甲" not in t.split(" ")[0] and "#" not in t and "`" not in t
    assert len(t) <= 50


def test_split_host_lines_variants():
    text = "甲：开场白\n乙：回应\n这是乙的补充\n还有更多补充"
    lines = CS._split_host_lines(text)
    assert [l["host"] for l in lines] == ["甲", "乙"]
    assert lines[1]["text"] == "回应 这是乙的补充 还有更多补充"   # 无前缀行并入上一条
    assert CS._split_host_lines("无前缀开头")[0]["host"] == "甲"   # 开头无前缀默认甲


def test_brief_line_injection():
    assert CS._brief_line("女主叫小七") == "\n自定义要求（必须遵守）：女主叫小七"
    assert CS._brief_line("") == "" and CS._brief_line(None) == "" and CS._brief_line("  ") == ""


def test_pipeline_llm_with_brief(monkeypatch):
    """LLM 正常：五段都注入自定义要求 brief（真 LLM 生成，不套模板）。"""
    import scholarpulse.llm as L
    seen = []

    class FakeLLM:
        def chat(self, sys, user, **kw):
            seen.append(user)
            return "【场景1 · 教室】\n小七：开拍！\n\n【场景2 · 操场】\n小七：收工！"
    monkeypatch.setattr(L, "get_llm", lambda: FakeLLM())
    o = CS.gen_outline("期末周的逆袭", genre="校园奇幻", brief="主角叫小七，热血风")
    assert o["mode"] == "llm" and "自定义要求（必须遵守）：主角叫小七，热血风" in seen[-1]
    assert "校园奇幻" in seen[-1]
    e = CS.expand_script("期末周的逆袭", o["text"], brief="主角叫小七")
    p = CS.polish_script("期末周的逆袭", e["text"], brief="主角叫小七")
    h = CS.dual_host_script("期末周的逆袭", p["text"], brief="主角叫小七")
    po = CS.post_production("期末周的逆袭", p["text"], brief="主角叫小七")
    assert all("自定义要求（必须遵守）：主角叫小七" in u for u in seen)   # 五段全部注入
    assert h["mode"] == "llm" and h["lines"] and h["speak"]
    assert not re.search(r"^甲[:：]", h["speak"], re.M)          # 朗读文本无"甲："前缀


def test_pipeline_failure_returns_error_no_template(monkeypatch):
    """LLM 全挂：返回 error 交给前端提示重试——绝不用降级模板假生成。"""
    import scholarpulse.llm as L

    def boom():
        raise RuntimeError("all providers down")
    monkeypatch.setattr(L, "get_llm", boom)
    for fn, args in [(CS.gen_outline, ("期末周的逆袭",)),
                     (CS.expand_script, ("期末周的逆袭", "大纲")),
                     (CS.polish_script, ("期末周的逆袭", "剧本")),
                     (CS.dual_host_script, ("期末周的逆袭", "剧本")),
                     (CS.post_production, ("期末周的逆袭", "剧本"))]:
        r = fn(*args, brief="热血风")
        assert "error" in r and "生成失败" in r["error"] and "大模型链路不可用" in r["error"]
        assert "text" not in r and "mode" not in r          # 没有模板产物


def test_split_scenes_blocks():
    """场景切分：按【场景…】切块 / 空行分段 / 上限截断。"""
    s1 = "【场景1 · 教室】\n老师：上课\n小七：到\n\n【场景2 · 操场】\n小七：跑\n旁白：风"
    sc = CS.split_scenes(s1)
    assert [x["title"] for x in sc] == ["场景1 · 教室", "场景2 · 操场"]
    assert sc[0]["lines"] == ["老师：上课", "小七：到"]
    s2 = "第一段台词\n\n第二段台词"
    assert len(CS.split_scenes(s2)) == 2                    # 空行分段
    s3 = "\n".join(f"【场景{i} · 地点】\n台词{i}" for i in range(1, 9))
    assert len(CS.split_scenes(s3, max_scenes=6)) == 6      # 上限截断
    assert CS.split_scenes("") == []                        # 空剧本
    s4 = "单行无标记剧本"
    assert CS.split_scenes(s4)[0]["lines"] == ["单行无标记剧本"]  # 单段兜底不丢内容


def test_export_creative_formats_and_invalid():
    args = ("测试短剧", OUTLINE_DEMO, "剧本正文", "甲：口播\n乙：接话", "【分镜】1) 特写")
    r_md = CS.export_creative(*args, fmt="md")
    assert r_md["size"] > 100 and r_md["path"].endswith(".md")
    assert "短剧大纲" in open(r_md["path"], encoding="utf-8").read()
    r_html = CS.export_creative(*args, fmt="html")
    html = open(r_html["path"], encoding="utf-8").read()
    assert "<h2>一、短剧大纲</h2>" in html and "pre" in html
    r_txt = CS.export_creative(*args, fmt="txt")
    txt = open(r_txt["path"], encoding="utf-8").read()
    assert txt.startswith("创意脚本包") and "双主持口播脚本" in txt
    bad = CS.export_creative(*args, fmt="docx")
    assert "error" in bad
