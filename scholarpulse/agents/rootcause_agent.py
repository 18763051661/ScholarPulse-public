"""根因追溯智能体：沿前置依赖图回溯到最底层根薄弱点。"""
from ..llm import get_llm

SYS = "你是课程顾问。根据前置依赖链，指出学生最该先补的根概念。输出JSON:{\"root\":\"...\",\"why\":\"...\"}"

def trace(chain: list[str]) -> dict:
    try:
        return get_llm().chat_json(SYS, f"依赖链:{' <- '.join(chain)}")
    except Exception:
        return {"root": chain[-1] if chain else "", "why": "前置不会则后置学不动"}
