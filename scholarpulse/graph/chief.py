"""
【总监副官 Chief-of-Staff】—— 多智能体三属性之一：自主性
与"固定流程编排"（hierarchy/policy_sandbox 的硬编码图）相对，副官是一个
ReAct 自主智能体：模型自己决定调哪个工具、调几次、何时收口。
- 复用 RouterChatModel.bind_tools（JSON 工具协议）→ 标准 tool_calls
- 复用 langgraph.prebuilt.create_react_agent 官方 ReAct 循环 + ToolNode
- 工具面 = tools_registry.ALL_TOOLS 全部 14 个（从单生辅导到学区治理沙盘）
返回 {steps:[{tool,args,result}], answer}：完整工具调用轨迹 + 最终答复，
供驾驶舱第⑤面板呈现"副官的思考路径"。
"""
from __future__ import annotations
from typing import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ..lc_model import RouterChatModel
from ..tools_registry import ALL_TOOLS

SYS_CHIEF = """你是学区总监的总督学副官。你可以调用工具来完成总监布置的任务，
覆盖：单生学情（query_weak_points 查薄弱考点 / make_quiz / cognitive_diagnose /
generate_report、build_twin_profile）、教学法（feynman_check/socratic_probe/
diagnose_error/trace_root_cause 需传概念依赖链 chain 如 ["一元二次方程","函数"] /
recommend_review/write_motivation_letter/make_flashcards）、学区治理
（seed_society_tool 合成人口 / run_abm 推进仿真 / governance_analyze 教师画像深析 /
epidemic_dashboard 误区疫情 / policy_sandbox 政策沙盘 / policy_committee 听证会）。
原则：先查数据再下结论；涉及学区级问题先确认已 seed/simulate；
拿到足够证据后给最终回答，不要无意义地反复调用工具。"""

# 教师端角色：同一工具池，不同的职责口吻——面向日常教学而非学区治理
SYS_TEACHER = """你是一线教师的教学助手。你拥有班级级诊断、单生学情、教学法三类工具，
遵循三步诊断 SOP：
① 群体问题（"班上/大家/整体"）→ 先 class_weak_map 扫共性薄弱考点，
   需要点名掉队学生时再 class_risk_scan；
② 个体问题 → query_weak_points 查该生薄弱点（传学号如 s001）；
③ 干预建议 → 结合 recommend_review / make_quiz_for_student 为最弱考点出分层题 /
   make_flashcards / write_motivation_letter 产出可执行方案。
其余：cognitive_diagnose 认知断层问诊 / diagnose_error 错因分类 /
trace_root_cause 追溯知识缺口根因（chain 参数必须传概念名列表而非学号，
可先查到薄弱概念再构造链）。
学区级工具（governance_analyze/epidemic_dashboard/policy_sandbox/policy_committee）
仅在教师明确询问年级/全校治理层面问题时才用。
铁律：先查数据再下结论；回答必须引用工具返回的具体数字（掌握度、人数、考点名），
建议落到具体考点、天数、做法；拿到足够证据后收口，不要无意义地反复调用工具。"""


class ChiefState(TypedDict, total=False):
    message: str
    steps: list
    answer: str


def _build_agent(sys_prompt: str = SYS_CHIEF):
    """create_react_agent（官方 ReAct 循环）；兼容新旧版 system 注入参数名。"""
    from langgraph.prebuilt import create_react_agent
    model = RouterChatModel(temperature=0.3, max_tokens=800).bind_tools(ALL_TOOLS)
    try:
        return create_react_agent(model, ALL_TOOLS, prompt=sys_prompt)
    except TypeError:
        return create_react_agent(model, ALL_TOOLS, state_modifier=sys_prompt)


def ask_chief(message: str, role: str = "chief",
              history: list | None = None) -> dict:
    """副官入口：自主 ReAct 多步工具调用，返回完整轨迹 + 最终答复。
    role="teacher" 时切换为教师端教学助手人设（同一工具池）。
    history 为 [(user, assistant)] 历史轮次：重放为无 tool_calls 的纯文本消息，
    使多轮追问（"那给 s001 出道题"）能继承上文。"""
    sys_prompt = SYS_TEACHER if role == "teacher" else SYS_CHIEF
    try:
        agent = _build_agent(sys_prompt)
        msgs: list = []
        for u, a in (history or [])[-6:]:
            msgs.append(HumanMessage(content=u))
            if a:
                msgs.append(AIMessage(content=a))
        msgs.append(HumanMessage(content=message))
        final = agent.invoke({"messages": msgs})
        steps, by_id, answer = [], {}, ""
        for m in final.get("messages", []):
            if isinstance(m, AIMessage):
                calls = getattr(m, "tool_calls", None) or []
                for c in calls:
                    s = {"tool": c["name"], "args": c.get("args") or {}, "result": ""}
                    steps.append(s)
                    by_id[c.get("id", "")] = s
                if not calls and m.content:      # 无工具调用的收口消息 = 最终回答
                    answer = m.content
            elif isinstance(m, ToolMessage):
                s = by_id.get(getattr(m, "tool_call_id", ""))
                if s is not None:
                    s["result"] = str(m.content)[:500]
        return {"agent": "ChiefOfStaff-ReAct", "steps": steps, "answer": answer}
    except Exception as e:                        # 全降级：至少给出可用的回答
        try:
            from ..llm import get_llm
            fb = get_llm().chat("你是学区总督学副官。" if role != "teacher"
                                else "你是一线教师的教学助手。", message)
        except Exception:
            fb = "助手暂时离线，请稍后再试。"
        return {"agent": "ChiefOfStaff-Fallback", "steps": [],
                "answer": fb, "error": str(e)}
