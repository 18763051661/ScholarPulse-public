# ScholarPulse 学脉 · 多智能体大学学情陪伴系统

> 用 LangChain / LangGraph 思路搭建的**六智能体自适应闭环**。不是"上传资料→向量检索→出套题"的套壳 RAG，而是把学情建模成一张**可演化的知识图谱**，由掌握度概率驱动下一轮任务，并内置**反幻觉校验**自动丢弃"讲义里根本不存在"的题目。

已实测跑通：DeepSeek / 通义千问 / 豆包(火山方舟) 三路故障转移，真实 LLM 调用。

---

## 一、为什么说它"创新"（和市面方案的区别）

| 维度 | 常见课件App | ScholarPulse |
|---|---|---|
| 知识表示 | 文本切块 → 向量库，扁平 | **知识图谱**：概念节点 + 前置依赖边 + 考试权重信号 |
| 学习状态 | 做对/做错计数 | **贝叶斯掌握度 p∈[0,1]**，逐题更新，可纵向对比 |
| 出题可信度 | LLM 直接出，常编造考点 | **反幻觉校验智能体**：第二智能体逐条核对"能否从上传讲义推出"，不通过即丢弃 |
| 任务推送 | 千人一面 / 随机 | **路径规划智能体**：只推"你薄弱 ∧ 高分考点 ∧ 是其他考点前置"的概念，并按学生分层 |
| 15天报表 | 流水账总结 | **纵向快照对比**：进步/停滞/考试风险，外加教师端班级共性薄弱与介入名单 |
| 垃圾知识点 | 无 | **噪声过滤智能体**：把内容分为 核心考点/了解即可/广告冗余，垃圾不进学习路径 |

### 六智能体闭环（`scholarpulse/agents/`）

```
课件 ─► [1.知识图谱构建] ─► [2.噪声过滤] ─► 知识图谱(concepts.json)
                                              │
              ┌───────────────────────────────┘
              ▼
      [5.路径规划] ─► [3.分层出题] ─► [4.反幻觉校验] ─► 题目(grounded=true)
              ▲                                            │
              │                                            ▼
              └──── [贝叶斯掌握度更新] ◄──── 学生答题 ◄────┘
                     （按更新后的掌握度决定是否进入下一轮）
                                              │
                                              ▼
                              [6.15天报表 + 教师班级分析]
```

闭环不是一次性：掌握度更新后若仍有薄弱高分考点，自动进入下一轮（轮次上限可配）。

---

## 二、工程结构

```
ScholarPulse/
├── run.py                     # CLI 入口（VSCode/IDEA 终端直接跑）
├── requirements.txt
├── .env                       # 三路密钥（已 gitignore，勿提交）
├── data/
│   └── sample_lecture.txt     # 示例课件（操作系统 第2章）
├── scholarpulse/
│   ├── config.py             # 解析 .env：供应商顺序/超时/温度/业务参数
│   ├── llm.py                 # 多供应商故障转移 + 容错JSON解析（截断自愈）
│   ├── models.py             # Pydantic：Concept/Mastery/QuizItem/...
│   ├── storage.py            # JSON 文件存储（IDE 里可直接打开看数据）
│   ├── mastery.py            # 贝叶斯掌握度引擎（纯计算，不调LLM）
│   ├── graph.py              # LangGraph StateGraph 编排
│   ├── report_html.py        # 渲染单文件 HTML 报告
│   └── agents/
│       ├── ingest_agent.py   # 1 知识图谱构建
│       ├── filter_agent.py   # 2 噪声过滤
│       ├── quiz_agent.py     # 3 分层出题
│       ├── verify_agent.py    # 4 反幻觉校验
│       ├── planner_agent.py  # 5 个性化路径规划
│       └── report_agent.py   # 6 15天报表 + 教师看板
└── output/                   # 生成的 HTML 报告
```

---

## 三、运行（VSCode / IDEA 终端）

```bash
cd ScholarPulse
pip install -r requirements.txt

# 1. 上传示例课件，构建知识图谱 + 噪声过滤
python run.py ingest

# 2. 跑一轮自适应学习闭环（--simulate 模拟学生答题，便于演示）
python run.py learn --sid s001 --simulate
#   真实使用去掉 --simulate，会逐题等你作答

# 3. 生成该学生 15 天学情报告 + HTML
python run.py report --sid s001
#   打开 output/report_s001.html

# 4. 教师登录后端看班级看板
python run.py teacher --teacher T001
```

**多供应商故障转移**：`.env` 中 `PROVIDER_ORDER=deepseek,qwen,doubao`，任一供应商超时/报错自动切下一家；豆包已配置 `thinking=disabled` 关闭思维链以降延迟。

---

## 四、实测输出（本次运行）

- 知识图谱：从操作系统讲义抽出 8 个考点（进程/PCB/状态转换/线程/死锁/四条件/银行家算法…），带前置依赖与考试权重 0.90~0.95。
- 反幻觉校验：第 2 轮**真实丢弃 2 道讲义无依据的题**。
- 闭环：3 轮自适应，答错"死锁"后该考点掌握度 0.47→0.31，第 3 轮自动把死锁排回优先。
- 教师看板：识别出班级共性薄弱（PCB/线程/切换/死锁）与需介入学生。

> 说明：`report` 第一次是"首次建档"，跑两次（间隔即 15 天）后会自动做两次快照的纵向对比，输出进步/停滞/考试风险。

---

## 五、从原型到产品的路线图

- [ ] 接入 PDF/Word 真实课件解析（`pypdf`/`python-docx`）替代 txt
- [ ] 教师后端：FastAPI + 登录鉴权，把 `teacher_dashboard` 变成网页
- [ ] 学生端：前端做题界面，替代 CLI 的 input
- [ ] 掌握度升级为 **IRT 项目反应理论**，按题目难度建模
- [ ] 图谱推理：前置依赖断路检测（前置没学会就不推后置）
- [ ] 报表定时任务：每 15 天自动生成并推送给指导老师编号
