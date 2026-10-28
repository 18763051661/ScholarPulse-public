"""全局配置：解析多供应商故障转移设置与业务参数。"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# 常见 OpenAI 兼容供应商预设：(端点, 默认模型)
# .env 里只要写 <前缀>_KEY 即可启用，端点和模型可被 <前缀>_BASE / <前缀>_MODEL 覆盖
PRESETS: dict[str, tuple[str, str]] = {
    "DSK": ("https://api.deepseek.com/chat/completions", "deepseek-chat"),
    "SILICONFLOW": ("https://api.siliconflow.cn/v1/chat/completions", "deepseek-ai/DeepSeek-V3"),
    "QWEN": ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen-turbo"),
    "DOUBAO": ("https://ark.cn-beijing.volces.com/api/v3/chat/completions", "doubao-seed-1-6-250615"),
    "KIMI": ("https://api.moonshot.cn/v1/chat/completions", "moonshot-v1-8k"),
    "ZHIPU": ("https://open.bigmodel.cn/api/paas/v4/chat/completions", "glm-4-flash"),
    "OPENROUTER": ("https://openrouter.ai/api/v1/chat/completions", "deepseek/deepseek-chat"),
    "GROQ": ("https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
    "MINIMAX": ("https://api.minimax.chat/v1/text/chatcompletion_v2", "abab6.5s-chat"),
    "BAICHUAN": ("https://api.baichuan-ai.com/v1/chat/completions", "Baichuan4"),
    "OLLAMA": ("http://127.0.0.1:11434/v1/chat/completions", "qwen2.5:7b"),
}

_RUNTIME_PATH = os.path.join("data", "llm_providers.json")


@dataclass
class Provider:
    name: str
    base: str          # 完整的 chat/completions 端点
    key: str
    model: str
    timeout_ms: int
    no_thinking: bool = False


def _get_provider(prefix: str) -> Provider | None:
    preset = PRESETS.get(prefix.upper(), ("", ""))
    base = os.getenv(f"{prefix}_BASE", "").strip() or preset[0]
    key = os.getenv(f"{prefix}_KEY", "").strip()
    model = os.getenv(f"{prefix}_MODEL", "").strip() or preset[1]
    # 本地 Ollama 不校验密钥
    if not (base and model) or not (key or prefix.upper() == "OLLAMA"):
        return None
    return Provider(
        name=prefix.lower(),
        base=base,
        key=key or "ollama",
        model=model,
        timeout_ms=int(os.getenv(f"{prefix}_TIMEOUT_MS", "10000")),
        no_thinking=os.getenv(f"{prefix}_NO_THINKING", "false").lower() == "true",
    )


def load_runtime_providers() -> list[dict]:
    """运行期在配置页添加的供应商（含密钥，落盘 data/llm_providers.json）。"""
    try:
        return json.load(open(_RUNTIME_PATH, encoding="utf-8")).get("providers", [])
    except Exception:
        return []


def save_runtime_providers(items: list[dict]) -> None:
    os.makedirs("data", exist_ok=True)
    try:
        json.dump({"providers": items}, open(_RUNTIME_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False)
    except Exception:
        pass


class Settings:
    # 解析有序供应商列表
    _order = [p.strip().upper() for p in os.getenv("PROVIDER_ORDER", "deepseek,qwen,doubao").split(",")]
    _prefix_map = {"DEEPSEEK": "DSK", "QWEN": "QWEN", "DOUBAO": "DOUBAO"}

    providers: list[Provider] = []
    runtime_names: set[str] = set()
    for alias in _order:
        p = _get_provider(_prefix_map.get(alias, alias))
        if p is not None:
            providers.append(p)
    # 追加运行期添加的自定义供应商
    for item in load_runtime_providers():
        try:
            providers.append(Provider(name=item["name"], base=item["base"],
                                      key=item["key"], model=item["model"],
                                      timeout_ms=int(item.get("timeout_ms", 10000)),
                                      no_thinking=bool(item.get("no_thinking"))))
            runtime_names.add(item["name"])
        except Exception:
            continue

    # 配置的原始链路顺序（切换模型时作为备用顺序的基准，避免越切越乱）
    base_names: list[str] = [p.name for p in providers]

    # 全局生成参数
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.6"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "600"))

    # 业务参数
    max_quiz_per_round: int = int(os.getenv("MAX_QUIZ_PER_ROUND", "4"))
    weak_threshold: float = float(os.getenv("WEAK_THRESHOLD", "0.6"))
    max_loop_rounds: int = int(os.getenv("MAX_LOOP_ROUNDS", "3"))
    report_interval_days: int = int(os.getenv("REPORT_INTERVAL_DAYS", "15"))

    @property
    def use_real_llm(self) -> bool:
        return len(self.providers) > 0


settings = Settings()


def add_runtime_provider(name: str, base: str, model: str, key: str,
                         timeout_ms: int = 10000) -> Provider:
    """配置页新增供应商：立即并入当前进程的故障转移链，并落盘。"""
    name = (name or "").strip().lower()
    if not name:
        raise ValueError("名称不能为空")
    if any(p.name == name for p in settings.providers):
        raise ValueError(f"供应商 {name} 已存在")
    if not (base.startswith("http") and model and key):
        raise ValueError("端点、模型、密钥都不能为空")
    p = Provider(name=name, base=base.strip(), key=key.strip(), model=model.strip(),
                 timeout_ms=int(timeout_ms or 10000))
    settings.providers.append(p)
    settings.runtime_names.add(name)
    settings.base_names.append(name)
    items = [x for x in load_runtime_providers() if x.get("name") != name]
    items.append({"name": p.name, "base": p.base, "model": p.model, "key": p.key,
                  "timeout_ms": p.timeout_ms})
    save_runtime_providers(items)
    return p


def remove_runtime_provider(name: str) -> None:
    name = (name or "").strip().lower()
    if name not in settings.runtime_names:
        raise ValueError("只能删除在配置页添加的模型")
    settings.providers[:] = [p for p in settings.providers if p.name != name]
    settings.runtime_names.discard(name)
    settings.base_names[:] = [n for n in settings.base_names if n != name]
    save_runtime_providers([x for x in load_runtime_providers() if x.get("name") != name])

