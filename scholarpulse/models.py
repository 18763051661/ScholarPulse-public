"""核心数据模型：知识图谱节点、掌握度、题目、报表。"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Concept(BaseModel):
    """知识图谱节点：从课件中抽取的一个考点概念。"""
    cid: str                       # 稳定 ID，如 "proc_deadlock"
    name: str                     # 概念名，如 "死锁"
    course: str = "os"             # 所属课程代号
    chapter: str = ""             # 所属章节
    summary: str = ""              # 一句话定义
    prerequisites: list[str] = Field(default_factory=list)   # 前置依赖 cid
    exam_weight: float = 0.5      # 考试权重 0~1（越高越可能考）
    tier: str = "core"             # foundational / core / advanced
    noise: str = "core"            # core(核心考点) / optional(了解) / filtered(垃圾冗余)


class QuizItem(BaseModel):
    """一道自测简答题。"""
    qid: str
    cid: str                       # 关联概念
    course: str = "os"             # 所属课程代号
    tier: str                      # 这道题的难度层
    question: str
    reference_answer: str           # 参考答案
    source_quote: str = ""         # 校验通过：来自讲义的原文摘录（反幻觉证据）
    grounded: bool = False          # 是否通过反幻觉校验


class Mastery(BaseModel):
    """某个学生对某个概念的掌握度（贝叶斯概率）。"""
    cid: str
    p: float = 0.3                 # 先验 0.3
    n_correct: int = 0
    n_wrong: int = 0
    updated_at: str = ""


class Student(BaseModel):
    sid: str
    name: str
    teacher_id: str = ""           # 唯一编号指定的指导老师
    tier: str = "foundation"       # foundation / consolidation / extension
    joined_at: str = ""


class AnswerRecord(BaseModel):
    qid: str
    cid: str
    correct: bool
    ts: str
    question: str = ""              # 题干快照（元认知审讯官的原料）
    reference: str = ""             # 参考答案快照


class ReportSnapshot(BaseModel):
    """一次报表时的掌握度快照，用于纵向对比。"""
    ts: str
    mastery: dict[str, float] = Field(default_factory=dict)


# ---------- 数字孪生对抗赛 ----------
class TwinMisconception(BaseModel):
    """孪生画像中的一条典型错误模式（认知误区）。"""
    cid: str
    name: str
    pattern: str                   # 误区描述，如"认为优先级越高越不可能饿死"


class TwinProfile(BaseModel):
    """学生数字孪生画像：LLM 依据答题历史推断的人格化模型。"""
    sid: str
    persona: str                   # 人格化描述
    misconceptions: list[TwinMisconception] = Field(default_factory=list)
    confidence_bias: str = "balanced"   # overconfident / underconfident / balanced
    similarity: float = -1         # 孪生与学生结果一致率 0~1，-1 表示尚无数据
    base_n: int = 0                # 建画像时的答题记录数，用于判断何时进化重建
    updated_at: str = ""


# ---------- 学区级多智能体治理：L0-L4 ----------
class Trait(BaseModel):
    """合成学生的潜在行为特质（ABM 驱动参数）。"""
    ability_prior: float = 0.5        # 能力先验 0~1
    procrastination: float = 0.3      # 拖延系数 λ：越高越少学习
    confidence_bias: str = "balanced" # overconfident / underconfident / balanced
    forget_rate: float = 0.05         # 艾宾浩斯遗忘率
    social_activity: float = 0.5      # 同伴互动概率（误区传染强度）


class Teacher(BaseModel):
    """指导教师：唯一编号 T001..T080。"""
    tid: str
    name: str
    subject: str = "操作系统"
    group_size: int = 4               # 每班学生规模基准


class SocietyEdge(BaseModel):
    """社交图一条边：学习小组内强连 或 跨班弱连接。"""
    src: str
    dst: str
    weight: float = 0.5
    kind: str = "study_group"         # study_group / weak_tie


class Society(BaseModel):
    """合成社会全量档案：特质 + 社交边 + 小组划分。"""
    seed: int
    traits: dict[str, Trait] = Field(default_factory=dict)
    edges: list[SocietyEdge] = Field(default_factory=list)
    groups: list[list[str]] = Field(default_factory=list)   # 学习小组（3-5人）
    created_at: str = ""


class ClassBrief(BaseModel):
    """教师智能体输出的班级画像（80 份 Map-Reduce 的 Map 产物）。"""
    tid: str
    n_students: int = 0
    avg_mastery: float = 0.0
    at_risk: list[str] = Field(default_factory=list)        # 高危学生 sid
    weak_concepts: list[str] = Field(default_factory=list)  # 班级共性薄弱考点
    risk_score: float = 0.0           # 0~1，越高越危险
    llm_analysis: str = ""            # 仅 Top-K 风险班有 LLM 深析


class GovernanceReport(BaseModel):
    """学区总监归约后的全局治理报告。"""
    ts: str = ""
    briefs: list[ClassBrief] = Field(default_factory=list)
    global_report: str = ""
    llm_calls: int = 0
    top_k: int = 5
    critique: str = ""   # Reflexion 评审官的批评意见（通过评审则留空）
    rounds: int = 1      # 报告生成轮数（director 被执行的次数）


class EpidemicState(BaseModel):
    """某个(学生,概念)的误区 SIR 状态。"""
    sid: str
    cid: str
    state: str = "S"                  # S 易感 / I 感染(持误区) / R 康复(已纠正)
    t_infect: int = -1                # 感染时刻（虚拟天）


class R0Estimate(BaseModel):
    """某个概念的误区基本传染数 R0 = β·k·D。"""
    cid: str
    beta: float = 0.0                 # 传播概率（每次接触）
    k: float = 0.0                    # 平均度
    d: float = 0.0                    # 平均传染期（天）
    r0: float = 0.0
    infected_n: int = 0


class PolicyExperiment(BaseModel):
    """一次反事实政策实验的完整记录。"""
    policy_text: str
    policy_json: dict = Field(default_factory=dict)
    days: int = 30
    seed: int = 42
    effect: dict = Field(default_factory=dict)   # {delta_avg, ci95, by_tier}
    narrative: str = ""
    risk_notes: str = ""


class DebateSpeech(BaseModel):
    """听证会中一位委员的一次发言。"""
    faction: str        # efficiency / equity / risk
    round: int          # 1 或 2
    speech: str = ""


class PolicyDebate(BaseModel):
    """政策听证委员会的完整记录：三派两轮交叉辩论 + 主持人裁决。"""
    ts: str = ""
    policy_text: str = ""
    speeches: list[DebateSpeech] = Field(default_factory=list)
    verdict: str = ""                     # 通过 / 有条件通过 / 暂缓
    rationale: str = ""                   # 裁决理由
    conditions: list[str] = Field(default_factory=list)   # 推行前置条件


class TwinBattle(BaseModel):
    """一场孪生对战的完整记录。"""
    bid: str
    qid: str
    cid: str
    question: str
    reference: str
    student_answer: str
    student_conf: int = 3
    twin_answer: str
    twin_conf: int = 3
    s_correct: bool = False
    t_correct: bool = False
    winner: str = "tie"            # student / twin / tie
    trick: str = ""                # 红方诱饵设计意图
    analysis: str = ""             # 揭晓时的偏差分析
    ts: str = ""
