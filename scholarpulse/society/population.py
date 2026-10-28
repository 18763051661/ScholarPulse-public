"""
【L0 人口合成智能体】—— 学区级治理的地基
合成一个可复现的教育社会：80 位教师、350 名学生、社交网络（班内学习小组强连 + 跨班弱连接）。
设计要点：
1. 全部随机经 random.Random(seed) 实例传递——同 seed 两次 seed_society 结果逐字节一致（政策沙盘反事实双跑的前提）
2. 合成学生 sid 强制 sim 前缀（Windows 文件系统大小写不敏感，避免与真实学生 s001 的 mastery_s001.json 冲突）
3. 同步把合成学生注册进 students.json，教师端 /api/teacher/class 与层级治理共享同一份人口
4. concepts.json 为空时内置兜底操作系统考点集（30 个，5 章，含先修关系与考权）
"""
from __future__ import annotations
import random
from ..models import Trait, Teacher, Student, Society, SocietyEdge, Concept
from .. import storage

N_TEACHERS = 80
N_STUDENTS = 350

SURNAMES = "王李张刘陈杨黄赵周吴徐孙马朱胡郭何高林罗"
GIVENS = ["伟", "芳", "娜", "敏", "静", "磊", "军", "洋", "勇", "艳", "杰", "涛", "明", "超",
          "秀英", "国强", "建华", "桂英", "晨", "雪", "琪", "宇", "浩", "欣怡", "子涵",
          "雨桐", "思远", "博文", "嘉懿", "诗涵", "煜城", "智渊", "泓涵", "芮涵", "星驰"]

# 兜底考点集：(cid, 名称, 章节, 考权, 层级, 先修cid列表, noise)
_CONCEPTS = [
    ("proc_state", "进程状态转换", "第一章 进程与线程", 0.9, "foundational", [], "core"),
    ("pcb", "进程控制块PCB", "第一章 进程与线程", 0.7, "foundational", ["proc_state"], "core"),
    ("thread_vs_proc", "线程与进程的区别", "第一章 进程与线程", 0.8, "core", ["proc_state"], "core"),
    ("context_switch", "上下文切换", "第一章 进程与线程", 0.6, "core", ["pcb"], "core"),
    ("sched_algo", "调度算法FCFS/SJF/RR", "第一章 进程与线程", 0.85, "core", ["proc_state"], "core"),
    ("priority_inversion", "优先级反转与饿死", "第一章 进程与线程", 0.5, "advanced", ["sched_algo"], "optional"),
    ("crit_section", "临界区与互斥", "第二章 同步与死锁", 0.9, "foundational", ["thread_vs_proc"], "core"),
    ("semaphore", "信号量机制", "第二章 同步与死锁", 0.9, "core", ["crit_section"], "core"),
    ("mutex", "互斥锁与自旋锁", "第二章 同步与死锁", 0.7, "core", ["crit_section"], "core"),
    ("producer_consumer", "生产者消费者问题", "第二章 同步与死锁", 0.85, "core", ["semaphore"], "core"),
    ("deadlock_cond", "死锁四个必要条件", "第二章 同步与死锁", 0.9, "core", ["crit_section"], "core"),
    ("banker_algo", "银行家算法", "第二章 同步与死锁", 0.8, "advanced", ["deadlock_cond"], "core"),
    ("deadlock_detect", "死锁检测与恢复", "第二章 同步与死锁", 0.6, "advanced", ["deadlock_cond"], "optional"),
    ("contig_alloc", "连续内存分配", "第三章 内存管理", 0.7, "foundational", [], "core"),
    ("paging", "分页存储管理", "第三章 内存管理", 0.9, "core", ["contig_alloc"], "core"),
    ("segmentation", "分段存储管理", "第三章 内存管理", 0.6, "core", ["contig_alloc"], "core"),
    ("page_replace", "页面置换算法LRU/FIFO", "第三章 内存管理", 0.85, "core", ["paging"], "core"),
    ("virtual_mem", "虚拟内存原理", "第三章 内存管理", 0.8, "core", ["paging"], "core"),
    ("thrashing", "抖动与工作集", "第三章 内存管理", 0.55, "advanced", ["virtual_mem"], "optional"),
    ("file_system", "文件系统结构", "第四章 文件与I/O", 0.7, "core", [], "core"),
    ("inode", "inode与硬链接", "第四章 文件与I/O", 0.6, "core", ["file_system"], "core"),
    ("dir_tree", "目录树与路径解析", "第四章 文件与I/O", 0.5, "foundational", ["file_system"], "core"),
    ("disk_sched", "磁盘调度算法", "第四章 文件与I/O", 0.65, "core", [], "core"),
    ("raid", "RAID级别", "第四章 文件与I/O", 0.45, "advanced", ["disk_sched"], "optional"),
    ("reader_writer", "读者写者问题", "第五章 经典问题与系统调用", 0.8, "advanced", ["semaphore"], "core"),
    ("pipe_comm", "管道通信", "第五章 经典问题与系统调用", 0.6, "core", ["proc_state"], "core"),
    ("signal_mech", "信号机制", "第五章 经典问题与系统调用", 0.5, "advanced", ["pipe_comm"], "optional"),
    ("syscall", "系统调用原理", "第五章 经典问题与系统调用", 0.75, "foundational", [], "core"),
    ("kernel_mode", "内核态与用户态", "第五章 经典问题与系统调用", 0.8, "foundational", ["syscall"], "core"),
    ("interrupt", "中断处理流程", "第五章 经典问题与系统调用", 0.7, "core", ["kernel_mode"], "core"),
]


def _ensure_concepts() -> list[Concept]:
    """concepts.json 为空时写入内置 OS 考点集（ABM/流行病学都依赖考点）。"""
    concepts = storage.load_concepts()
    if concepts:
        return concepts
    concepts = [Concept(cid=cid, name=name, chapter=ch, summary=f"{name}的核心考点与典型易错点",
                        prerequisites=pre, exam_weight=w, tier=tier, noise=noise)
                for cid, name, ch, w, tier, pre, noise in _CONCEPTS]
    storage.save_concepts(concepts)
    return concepts


def _gen_teachers(rng: random.Random) -> dict[str, Teacher]:
    teachers = {}
    for i in range(1, N_TEACHERS + 1):
        tid = f"T{i:03d}"
        teachers[tid] = Teacher(tid=tid, name=rng.choice(SURNAMES) + rng.choice(GIVENS),
                                group_size=rng.choice([4, 4, 4, 5]))
    return teachers


def _gen_students(rng: random.Random, teachers: dict[str, Teacher]) -> tuple[dict[str, Student], dict[str, Trait], dict[str, list[str]]]:
    """返回 (students, traits, tid→sids 班级名册)。350 = 30 班×5 + 50 班×4。"""
    students: dict[str, Student] = {}
    traits: dict[str, Trait] = {}
    roster: dict[str, list[str]] = {tid: [] for tid in teachers}
    # 30 个班 5 人、50 个班 4 人
    quota = [5] * 30 + [4] * 50
    rng.shuffle(quota)
    tids = list(teachers)
    idx = 0
    now = storage.now_iso()
    for i in range(1, N_STUDENTS + 1):
        while quota[idx] == 0:
            idx += 1
        tid = tids[idx]
        quota[idx] -= 1
        sid = f"sim{i:03d}"
        students[sid] = Student(sid=sid, name=rng.choice(SURNAMES) + rng.choice(GIVENS),
                                teacher_id=tid, tier="foundation", joined_at=now)
        traits[sid] = Trait(
            ability_prior=min(0.9, max(0.1, rng.gauss(0.5, 0.15))),
            procrastination=round(rng.uniform(0.05, 0.8), 3),
            confidence_bias=rng.choice(["overconfident", "underconfident", "balanced"]),
            forget_rate=round(min(0.08, max(0.005, rng.gauss(0.03, 0.015))), 4),
            social_activity=round(rng.uniform(0.1, 0.9), 3),
        )
        roster[tid].append(sid)
    return students, traits, roster


def _build_groups(rng: random.Random, roster: dict[str, list[str]]) -> list[list[str]]:
    """每个班内部划 3-5 人学习小组（余数并入最后一组）。"""
    groups: list[list[str]] = []
    for tid, sids in roster.items():
        pool = sids[:]
        rng.shuffle(pool)
        while len(pool) >= 3:
            n = rng.choice([3, 4, 5])
            groups.append(sorted(pool[:n]))
            pool = pool[n:]
        if pool:
            if groups and len(groups[-1]) + len(pool) <= 6:
                groups[-1] = sorted(groups[-1] + pool)   # 零头并入班内上一组
            else:
                groups.append(sorted(pool))
    return groups


def _build_edges(rng: random.Random, groups: list[list[str]],
                 roster: dict[str, list[str]]) -> list[SocietyEdge]:
    edges: list[SocietyEdge] = []
    seen: set[tuple[str, str]] = set()

    def add(a: str, b: str, w: float, kind: str):
        key = (a, b) if a < b else (b, a)
        if a == b or key in seen:
            return
        seen.add(key)
        edges.append(SocietyEdge(src=key[0], dst=key[1], weight=w, kind=kind))

    for g in groups:                       # 组内团（完全子图，强连接）
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                add(g[i], g[j], 0.6, "study_group")
    all_sids = sorted({s for g in groups for s in g})
    class_of = {sid: tid for tid, sids in roster.items() for sid in sids}
    n_weak = int(len(all_sids) * 0.10)     # 10% 跨班弱连接
    for _ in range(n_weak * 4):
        if sum(1 for e in edges if e.kind == "weak_tie") >= n_weak:
            break
        a, b = rng.sample(all_sids, 2)
        if class_of[a] != class_of[b]:
            add(a, b, 0.15, "weak_tie")
    return edges


def seed_society(seed: int = 42) -> dict:
    """合成 80 教师 + 350 学生的教育社会并落盘。同 seed 严格可复现。"""
    rng = random.Random(seed)
    concepts = _ensure_concepts()
    teachers = _gen_teachers(rng)
    students, traits, roster = _gen_students(rng, teachers)
    groups = _build_groups(rng, roster)
    edges = _build_edges(rng, groups, roster)
    society = Society(seed=seed, traits=traits, edges=edges, groups=groups,
                      created_at=storage.now_iso())
    # 合成学生注册进总名册（保留真实学生 s001 不动）
    all_students = storage.load_students()
    all_students.update(students)
    storage.save_students(all_students)
    storage.save_teachers(teachers)
    storage.save_society(society)
    return {"teachers": len(teachers), "students": len(students),
            "concepts": len(concepts), "edges": len(edges), "groups": len(groups)}


# ---------- 纯函数（可单测） ----------
def class_of(society: Society, students: dict[str, Student]) -> dict[str, list[str]]:
    """按 teacher_id 归组班级名册（含真实学生）。"""
    roster: dict[str, list[str]] = {}
    for sid, s in students.items():
        roster.setdefault(s.teacher_id, []).append(sid)
    return roster
