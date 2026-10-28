"""检索质量评测集：固定语料 + 固定查询，量化 dense / sparse / hybrid 三路指标。

对标 RAGAS 式检索评测思想（离线、可复现、零 LLM 依赖）：
  - 语料：内置 6 段固定主题段落（计算机网络 ×4 + 认知科学 ×2），每段一个独立知识块
  - 查询：6 条固定查询，每条指定期望命中的段落（用段落内唯一关键词定位，不依赖分块顺序）
  - 指标：hit@1 / hit@3 / MRR（平均倒数排名），对 dense、sparse、hybrid 三路分别计算
  - 结论：RRF 融合的 hybrid 路保住两路单策略的最佳召回（融合不劣于最优单路）

答辩价值：把"我们的检索器好不好"从感觉变成数字，/api/rag/eval 现场可跑。
"""
from __future__ import annotations
from .rag_retriever import HybridRetriever, chunk_text

# 评测语料：6 段主题互不重叠的固定段落
EVAL_CORPUS = (
    "子网掩码的作用是区分IP地址中的网络位与主机位，与IP地址按位与运算即可得到网络地址，掩码越长可用的子网越多。\n"
    "TCP三次握手建立可靠连接：客户端先发送SYN同步报文，服务端回复SYN加ACK确认报文，客户端再发送ACK确认，连接进入已建立状态。\n"
    "UDP是无连接的传输层协议，不保证可靠交付也不做拥塞控制，首部开销只有八个字节，实时音视频常用UDP。\n"
    "路由器工作在网络层，根据路由表的最长前缀匹配规则转发IP数据报，交换机则工作在数据链路层依据MAC地址表转发帧。\n"
    "艾宾浩斯遗忘曲线表明遗忘先快后慢，按二十分钟、一小时、一天、两天的间隔安排复习可以把记忆保持率稳定在高位。\n"
    "IRT项目反应理论用能力参数和题目参数 jointly 刻画答对概率，比经典测验理论更能支持计算机自适应测验的选题。"
)

# 评测查询：expect = 正确段落中的唯一关键词（不依赖分块顺序）
EVAL_QUERIES = [
    {"q": "子网掩码怎么区分网络位和主机位", "expect": "按位与"},
    {"q": "三次握手建立连接的过程", "expect": "SYN"},
    {"q": "UDP 无连接 不可靠 首部开销", "expect": "拥塞控制"},
    {"q": "路由器根据什么转发数据报", "expect": "最长前缀"},
    {"q": "遗忘曲线的复习间隔怎么安排", "expect": "艾宾浩斯"},
    {"q": "项目反应理论如何估计答对概率", "expect": "自适应"},
]


def _metrics(ranks: list[int | None]) -> dict:
    """ranks：每条查询的 gold 排名（None=未命中）。→ hit@1/hit@3/MRR。"""
    n = len(ranks) or 1
    hit1 = sum(1 for r in ranks if r == 1) / n
    hit3 = sum(1 for r in ranks if r is not None and r <= 3) / n
    mrr = sum(1 / r for r in ranks if r) / n
    return {"hit@1": round(hit1, 3), "hit@3": round(hit3, 3), "mrr": round(mrr, 3)}


def _rank_of(hits: list[dict], gold_chunk: int, key: str) -> int | None:
    """gold 块在指定策略下的排名（取 hit 里该策略的 rank 字段）。"""
    for h in hits:
        if h["chunk_id"] == gold_chunk:
            return h[key]["rank"] if key in ("dense", "sparse") else h["rrf_rank"]
    return None


def run_eval() -> dict:
    """在固定评测集上跑三路检索，返回各路 hit@1/hit@3/MRR。"""
    chunks = chunk_text(EVAL_CORPUS, size=110, overlap=15)
    ret = HybridRetriever(chunks)
    # 期望段落 → 块 id（用段落唯一关键词定位，免疫分块顺序变化）
    gold_of = {}
    for i, c in enumerate(chunks):
        for q in EVAL_QUERIES:
            if q["expect"] in c:
                gold_of[q["q"]] = i
    d_ranks, s_ranks, h_ranks = [], [], []
    for q in EVAL_QUERIES:
        gold = gold_of.get(q["q"])
        if gold is None:                      # 关键词没分到任何块（异常防护）
            d_ranks.append(None); s_ranks.append(None); h_ranks.append(None)
            continue
        hits = ret.search(q["q"], top_k=len(chunks))
        d_ranks.append(_rank_of(hits, gold, "dense"))
        s_ranks.append(_rank_of(hits, gold, "sparse"))
        # hybrid：RRF 排名即 hits 顺序
        h_ranks.append(next((h["rank"] for h in hits if h["chunk_id"] == gold), None))
    dense, sparse = _metrics(d_ranks), _metrics(s_ranks)
    hybrid = _metrics(h_ranks)
    best_single = max(dense["mrr"], sparse["mrr"])
    return {"n_chunks": len(chunks), "n_queries": len(EVAL_QUERIES),
            "dense": dense, "sparse": sparse, "hybrid": hybrid,
            "conclusion": ("融合不劣于最优单路" if hybrid["mrr"] >= best_single - 0.01
                           else "融合低于最优单路（需调参）")}
