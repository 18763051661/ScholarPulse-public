"""监考裁决消融实验测试（全离线、固定种子可复现）：
多维交叉验证的误报率必须显著低于单维规则，且漏报率不劣化。"""
from scholarpulse.core.proctor_ablation import (
    CHEAT_KINDS,
    build_dataset,
    judge,
    run_experiment,
)


def test_dataset_composition():
    """40 份会话 = 30 正常（18 纯净 + 12 单维噪声）+ 10 作弊（4 类手法）。"""
    ds = build_dataset(seed=42)
    assert len(ds) == 40
    labels = [s["label"] for s in ds]
    assert labels.count("正常") == 30
    for k, n in (("背答案", 3), ("中途换人", 2), ("朗读答案", 2), ("离席查资料", 3)):
        assert labels.count(k) == n, f"作弊手法 {k} 数量应为 {n}"


def test_determinism_same_seed():
    """同种子两次实验结果完全一致（离线可复现）。"""
    a = run_experiment(seed=7)
    b = run_experiment(seed=7)
    assert a["baseline_single_dim"] == b["baseline_single_dim"]
    assert a["upgraded_cross_validated"] == b["upgraded_cross_validated"]
    assert [c["baseline_flag"] for c in a["cases"]] == \
           [c["baseline_flag"] for c in b["cases"]]


def test_headline_fpr_drops_fnr_kept():
    """核心结论：多维交叉误报率显著低于单维规则（0 vs ≥30%），漏报率同为 0。"""
    rep = run_experiment(seed=42)
    base, up = rep["baseline_single_dim"], rep["upgraded_cross_validated"]
    assert base["fpr"] >= 0.30, "单维规则应误伤至少 30% 正常考生"
    assert up["fpr"] == 0.0, "多维交叉不应误伤任何正常考生"
    assert base["fnr"] == 0.0 and up["fnr"] == 0.0, "两套裁决器都不应漏判作弊"
    assert up["f1"] > base["f1"]
    assert up["precision"] == 1.0


def test_every_cheat_kind_caught_by_cross_validate():
    """四类作弊手法全部被多维交叉验证/巡逻捕获（生产工具真实生效）。"""
    for kind in CHEAT_KINDS:
        hits = 0
        for s in build_dataset(seed=42):
            if s["label"] == kind:
                if judge(s)["upgraded_flag"]:
                    hits += 1
        assert hits >= 1, f"作弊手法「{kind}」未被多维交叉验证捕获"


def test_clean_normals_have_zero_flags():
    """18 份纯净正常会话在单维规则下也应零标记（噪声才是误报来源）。"""
    for s in build_dataset(seed=42):
        if s["kind"] == "正常":
            r = judge(s)
            assert not r["baseline_flag"], f"纯净考生被单维规则误伤: {r['single_flags']}"


def test_report_schema_and_rates_in_range():
    """输出结构完整、比率均在 [0,1]、headline 文案可生成。"""
    rep = run_experiment(seed=42)
    for key in ("dataset", "baseline_single_dim", "upgraded_cross_validated",
                "cheat_recall_by_kind", "headline", "cases"):
        assert key in rep
    for m in (rep["baseline_single_dim"], rep["upgraded_cross_validated"]):
        for k in ("fpr", "fnr", "precision", "recall", "f1"):
            assert 0.0 <= m[k] <= 1.0
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        assert tp + fp + tn + fn == 40
    assert rep["headline"]
