"""
自定义 LangChain ChatModel：RouterChatModel。
- 继承 langchain_core 的 BaseChatModel，可与 ChatPromptTemplate / JsonOutputParser / LangGraph 无缝组合
- 底层走多供应商 OpenAI 兼容接口故障转移（DeepSeek→千问→豆包）
- 支持 bind_tools()：把工具清单注入提示词并约定 JSON 协议
  （模型输出 {"tool":名,"args":{...}} 或 {"answer":"..."}），再合成标准 tool_calls 消息——
  于是 create_react_agent / ToolNode 等官方机制都能直接跑在我们自己的 HTTP 路由模型上。
课程报告亮点：不依赖单一厂商 SDK，而是把自有 HTTP 客户端包装成标准 LangChain 组件。
"""
from __future__ import annotations
import json
import uuid
from typing import Any, List, Optional, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (BaseMessage, HumanMessage, SystemMessage,
                                     AIMessage, ToolMessage)
from langchain_core.outputs import ChatGeneration, ChatResult
from .llm import get_llm

_ANSWER = "__ANSWER__"


class RouterChatModel(BaseChatModel):
    """一个可被 LangChain/LangGraph 调用的聊天模型，支持工具绑定（JSON 协议）。"""
    temperature: float = 0.6
    max_tokens: int = 600
    bound_tools: list = []  # bind_tools 注入的 OpenAI 工具 schema 列表

    @property
    def _llm_type(self) -> str:
        return "scholarpulse-router"

    # ---------- 工具绑定 ----------
    def bind_tools(self, tools: list, **kwargs: Any) -> "RouterChatModel":
        """把工具 schema 冻结进一个新模型实例（BaseChatModel 的标准协议）。"""
        schemas = []
        for t in tools:
            try:
                from langchain_core.utils.function_calling import convert_to_openai_tool
                schemas.append(convert_to_openai_tool(t))
            except Exception:
                name = getattr(t, "name", str(t))
                desc = getattr(t, "description", "")
                schemas.append({"type": "function",
                                "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": {}}}})
        return self.model_copy(update={"bound_tools": schemas})

    # ---------- 消息扁平化：把 ReAct 多轮消息压成纯文本对话 ----------
    @staticmethod
    def _flatten(messages: List[BaseMessage]) -> str:
        lines = []
        for m in messages:
            if isinstance(m, SystemMessage):
                lines.append(f"[系统] {m.content}")
            elif isinstance(m, HumanMessage):
                lines.append(f"[用户] {m.content}")
            elif isinstance(m, AIMessage):
                calls = getattr(m, "tool_calls", None)
                if calls:
                    brief = "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                                      for c in calls)
                    lines.append(f"[助手] 我决定调用工具: {brief}")
                elif m.content:
                    lines.append(f"[助手] {m.content}")
            elif isinstance(m, ToolMessage):
                lines.append(f"[工具 {m.name} 返回] {m.content}")
        return "\n".join(lines)

    # ---------- JSON 工具协议解析 ----------
    @staticmethod
    def _parse_tool_json(raw: str) -> Optional[Tuple[str, Any]]:
        """从模型输出抠出 {"tool","args"} / {"answer"}；返回 (工具名, args) 或
        (_ANSWER, 文本)；彻底失败返回 None（调用方降级为纯文本回答）。"""
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()
        start, end = s.find("{"), s.rfind("}")
        if start < 0:
            return None
        raw_obj = s[start:end + 1] if end > start else s[start:]  # 无闭合 } = 截断片段
        try:
            obj = json.loads(raw_obj)
        except Exception:
            # JSON 被截断（多为 max_tokens 截尾）→ 兜底抠出 answer 值前缀或工具名
            i = s.find('"answer"')
            if i >= 0:
                colon = s.find(":", i + 1)
                j = s.find('"', colon + 1)          # answer 值的起始引号
                if j >= 0:
                    val = s[j + 1:].rstrip().rstrip('"').rstrip()
                    if val:
                        return (_ANSWER, val)
            # tool JSON 截断：至少抠出工具名 + 空 args（工具报错会触发模型观察重试）
            i = s.find('"tool"')
            if i >= 0:
                colon = s.find(":", i + 1)
                j = s.find('"', colon + 1)
                k = s.find('"', j + 1)
                if j >= 0 and k > j and s[j + 1:k]:
                    return (s[j + 1:k], {})
            return None
        if not isinstance(obj, dict):
            return None
        if "answer" in obj and "tool" not in obj:
            return (_ANSWER, str(obj["answer"]))
        if "tool" in obj:
            args = obj.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return (str(obj["tool"]), args if isinstance(args, dict) else {})
        return None

    # ---------- 生成 ----------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "你是助手。")
        user = self._flatten(messages)

        # 有工具绑定：JSON 协议二选一（调工具 / 给最终回答）
        if self.bound_tools:
            menu = "\n".join(f"- {s['function']['name']}: {s['function'].get('description', '')}"
                             for s in self.bound_tools)
            sys2 = (system + "\n\n你可以调用以下工具：\n" + menu +
                    "\n严格只输出一个 JSON 对象，二选一："
                    '{"tool":"工具名","args":{参数}} 或 {"answer":"最终回答"}。不要输出其他文字。')
            raw = get_llm().chat(sys2, user, temperature=self.temperature, max_tokens=self.max_tokens)
            parsed = self._parse_tool_json(raw)
            if parsed is None:  # 解析失败 → 当最终回答，自然终止 ReAct 循环
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=raw))])
            name, payload = parsed
            if name == _ANSWER:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=payload))])
            msg = AIMessage(content="", tool_calls=[{"name": name, "args": payload,
                                                     "id": f"call_{uuid.uuid4().hex[:10]}"}])
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # 无工具绑定：普通对话
        text = get_llm().chat(system, user, temperature=self.temperature, max_tokens=self.max_tokens)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
