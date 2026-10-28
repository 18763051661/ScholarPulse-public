"""数字孪生对抗赛纯函数测试：胜负判定与相似度口径。"""
from scholarpulse.agents.twin_agent import judge_winner, calc_similarity
from scholarpulse.models import TwinBattle, TwinProfile, TwinMisconception


def _battle(winner: str, s: bool, t: bool, ts: str = "t") -> TwinBattle:
    return TwinBattle(bid="b", qid="q", cid="c", question="Q", reference="R",
                      student_answer="a", twin_answer="b", s_correct=s,
                      t_correct=t, winner=winner, ts=ts)


def test_judge_winner():
    assert judge_winner(True, False) == "student"
    assert judge_winner(False, True) == "twin"
    assert judge_winner(True, True) == "tie"
    assert judge_winner(False, False) == "tie"


def test_calc_similarity_empty():
    r = calc_similarity([])
    assert r["sim"] == -1.0
    assert r["sim_trend"] == []


def test_calc_similarity_full_match():
    hist = [_battle("tie", True, True), _battle("tie", False, False)]
    r = calc_similarity(hist)
    assert r["sim"] == 1.0
    assert r["sim_trend"][-1]["sim"] == 1.0


def test_calc_similarity_mixed_and_window():
    hist = [_battle("tie", True, True, "t1"),
            _battle("student", True, False, "t2"),
            _battle("twin", False, True, "t3")]
    r = calc_similarity(hist)
    assert r["sim"] == round(1 / 3, 2)
    # 滑窗：第2场窗口 [对对, 对错] -> 0.5；第3场窗口 [对对,对错,错对] -> 0.33
    assert r["sim_trend"][1]["sim"] == 0.5
    assert len(r["sim_trend"]) == 3


def test_models_roundtrip():
    p = TwinProfile(sid="s001", persona="测试画像",
                    misconceptions=[TwinMisconception(cid="c1", name="死锁", pattern="混淆必要条件")])
    assert p.similarity == -1 and p.base_n == 0
    b = _battle("student", True, False)
    assert b.winner == "student" and b.twin_conf == 3
