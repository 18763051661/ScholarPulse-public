"""GraphRAG 图增强 + 检索质量评测集测试（全离线）。"""
from types import SimpleNamespace as NS

from scholarpulse.core.rag_retriever import expand_with_graph
from scholarpulse.core.rag_eval import run_eval, EVAL_QUERIES

# 概念链：A ← B ← C ← D（B 的先修是 A，依此类推）
CONCEPTS = [NS(cid="ca", name="网络基础", chapter="第一章", prerequisites=[]),
            NS(cid="cb", name="子网掩码", chapter="第一章", prerequisites=["ca"]),
            NS(cid="cc", name="子网划分", chapter="第二章", prerequisites=["cb"]),
            NS(cid="cd", name="超网聚合", chapter="第二章", prerequisites=["cc"])]
MASTERY = {"ca": NS(p=0.3), "cb": NS(p=0.8)}


def test_graph_expansion_upstream_downstream():
    hits = [{"text": "子网掩码用于区分网络位与主机位", "rank": 1}]
    graph = expand_with_graph(hits, CONCEPTS, MASTERY)
    names = {(g["name"], g["relation"]) for g in graph}
    assert ("网络基础", "子网掩码的上游先修") in names      # 上游一跳
    assert ("子网划分", "子网掩码的下游依赖") in names      # 下游一跳
    assert all("超网聚合" not in g["name"] for g in graph)  # 只扩展一跳，不到两跳


def test_graph_mastery_attached():
    hits = [{"text": "子网掩码掩码越长子网越多", "rank": 1}]
    graph = expand_with_graph(hits, CONCEPTS, MASTERY)
    m = {g["name"]: g["mastery"] for g in graph}
    assert m["网络基础"] == 0.3                             # 带掌握度证据
    assert m["子网划分"] is None                            # 无掌握度数据的为 None


def test_graph_dedup_and_cap():
    hits = [{"text": "子网掩码与网络基础", "rank": 1}]       # 两个概念同现
    graph = expand_with_graph(hits, CONCEPTS, MASTERY, max_neighbors=2)
    assert len(graph) <= 2                                  # 截断生效
    assert len({g["cid"] for g in graph}) == len(graph)     # 去重


def test_eval_structure():
    r = run_eval()
    assert r["n_queries"] == len(EVAL_QUERIES) == 6
    for strat in ("dense", "sparse", "hybrid"):
        assert set(r[strat]) == {"hit@1", "hit@3", "mrr"}
    assert r["conclusion"] in ("融合不劣于最优单路", "融合低于最优单路（需调参）")


def test_eval_hybrid_preserves_best_recall():
    r = run_eval()
    best_single = max(r["dense"]["mrr"], r["sparse"]["mrr"])
    assert r["hybrid"]["mrr"] >= best_single - 0.01         # 融合不劣于最优单路
    assert r["hybrid"]["hit@3"] == 1.0                      # 固定评测集全命中
    assert r["dense"]["mrr"] > 0 and r["sparse"]["mrr"] > 0  # 两路都有效


def test_graph_expansion_with_real_concepts():
    """用真实概念库做集成冒烟：命中"视图"应扩展出其先修概念。"""
    from scholarpulse import storage
    concepts = storage.load_concepts()
    if not concepts:
        return
    target = next((c for c in concepts if c.name == "视图" and c.prerequisites), None)
    if not target:
        return
    hits = [{"text": f"{target.name}是基于基表导出的虚表", "rank": 1}]
    graph = expand_with_graph(hits, concepts)
    assert any(g["cid"] in target.prerequisites for g in graph)
