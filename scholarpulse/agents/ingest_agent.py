"""
【智能体1：知识图谱构建器】
创新点：不是把课件切块做 RAG，而是抽取「概念 + 前置依赖边 + 考试权重信号」。
输出结构化 JSON，后续所有个性化都跑在这张图上。
"""
from __future__ import annotations
from ..llm import get_llm
from ..models import Concept

SYSTEM = """你是大学课程知识图谱构建专家。从讲义中抽取考点概念，输出紧凑JSON。
要求：
1. 只抽取真正的知识点（名词/原理/算法），不要口号、目录、例题编号。
2. 每个概念给出前置依赖(prerequisites)：学好它必须先懂哪些概念的id。
3. exam_weight(0~1)：根据讲义中"重点/必考/核心/掌握"等信号判断考试可能性。
4. tier: foundational(基础概念) / core(核心考点) / advanced(拔高)。
输出格式: {"concepts":[{"cid":"英文短下划线id","name":"中文名","chapter":"章节","summary":"一句话定义","prerequisites":["cid"],"exam_weight":0.0,"tier":"core"}]}
最多8个概念，summary不超过30字。"""


def build_knowledge_graph(material_text: str, material_name: str) -> list[Concept]:
    user = f"讲义文件：{material_name}\n\n讲义内容：\n{material_text[:3500]}"
    data = get_llm().chat_json(SYSTEM, user)
    concepts = []
    for c in data.get("concepts", []):
        try:
            concepts.append(Concept(
                cid=c["cid"], name=c["name"], chapter=c.get("chapter", ""),
                summary=c.get("summary", ""), prerequisites=c.get("prerequisites", []),
                exam_weight=float(c.get("exam_weight", 0.5)), tier=c.get("tier", "core"),
            ))
        except Exception:
            continue
    return concepts
