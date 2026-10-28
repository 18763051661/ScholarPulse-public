"""多供应商配置与真实探针测试（全离线，HTTP 层被替身接管）：
- 切换语义：选中供应商提到链首，其余回到配置原始顺序，反复切换不越切越乱
- 探针语义：成功记 ok + token；失败记 fail 且不计 token（不能把失败算成消耗）
- 配置安全：密钥只出现掩码形态，完整 key 不得出现在返回体里
- 额度口径：有公开余额接口的查真实值，没有的明确标注 supported=False
"""
import json

import pytest

import scholarpulse.llm as llm
from scholarpulse.config import Provider, settings


class FakeResp:
    def __init__(self, payload=None, boom=None):
        self._payload = payload
        self._boom = boom

    def raise_for_status(self):
        if self._boom:
            raise self._boom

    def json(self):
        return self._payload


class FakeClient:
    """替身 HTTP 客户端：按预设返回 payload 或抛异常，并记录请求次数。"""

    calls: list = []

    def __init__(self, payload=None, boom=None):
        self.payload, self.boom = payload, boom

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, **kw):
        FakeClient.calls.append(("post", url))
        return FakeResp(self.payload, self.boom)

    def get(self, url, **kw):
        FakeClient.calls.append(("get", url))
        return FakeResp(self.payload, self.boom)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """独立状态文件 + 三家固定供应商，互不污染真实 data/llm_state.json。"""
    monkeypatch.setattr(llm, "_STATE_PATH", str(tmp_path / "llm_state.json"))
    provs = [Provider(name="a", base="https://a.example/v1/chat/completions",
                      key="sk-aaaaaaaaaaaa1111", model="ma", timeout_ms=3000),
             Provider(name="b", base="https://b.example/v1/chat/completions",
                      key="sk-bbbbbbbbbbbb2222", model="mb", timeout_ms=3000),
             Provider(name="c", base="https://c.example/v1/chat/completions",
                      key="sk-cccccccccccc3333", model="mc", timeout_ms=3000)]
    monkeypatch.setattr(settings, "providers", provs)
    monkeypatch.setattr(settings, "base_names", ["a", "b", "c"])
    FakeClient.calls = []
    return llm.LLMClient()


def test_switch_promotes_and_restores(client):
    assert client.config()["chain"] == ["a", "b", "c"]
    assert client.set_active("c")["chain"] == ["c", "a", "b"]   # 选中提到链首
    assert client.set_active("b")["chain"] == ["b", "a", "c"]   # 其余保持原始相对顺序
    assert client.set_active("a")["chain"] == ["a", "b", "c"]   # 反复切换可完全复位
    with pytest.raises(ValueError):
        client.set_active("nope")


def test_probe_success_records_tokens(client, monkeypatch):
    payload = {"choices": [{"message": {"content": "在线"}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    monkeypatch.setattr(llm.httpx, "Client", lambda **kw: FakeClient(payload))

    r = client.probe("b")
    assert r["ok"] and r["reply"] == "在线" and r["tokens"] == 15
    assert client.stats["b"] == {"ok": 1, "fail": 0}
    assert client.config()["providers"][1]["probe"]["ok"] is True
    assert client.usage()["total"]["req"] == 1          # 记账 +1
    assert client.usage()["total"]["tokens"] == 15


def test_probe_failure_does_not_charge(client, monkeypatch):
    monkeypatch.setattr(llm.httpx, "Client",
                        lambda **kw: FakeClient(boom=RuntimeError("connect timeout")))

    r = client.probe("c")
    assert r["ok"] is False and "timeout" in r["error"]
    assert client.stats["c"] == {"ok": 0, "fail": 1}
    assert client.usage()["total"]["req"] == 0           # 失败不计 token
    # 探针结果落盘：换一个实例读同一状态文件仍是失败记录
    again = llm.LLMClient()
    assert again.config()["providers"][2]["probe"]["ok"] is False


def test_config_masks_secret(client):
    cfg = client.config()
    body = json.dumps(cfg, ensure_ascii=False)
    for p in cfg["providers"]:
        assert "****" in p["key"] and len(p["key"]) < 16  # 只保留前5后4
    assert "sk-aaaaaaaaaaaa1111" not in body             # 完整密钥不外泄
    assert cfg["providers"][0]["key"] == "sk-aa****1111"


def test_quota_marks_unsupported(client, monkeypatch):
    out = client.quota()
    assert all(x["supported"] is False for x in out)     # 三家都不是 deepseek
    # deepseek 端点走真实余额接口（HTTP 仍为替身）
    client.providers[0].base = "https://api.deepseek.com/chat/completions"
    monkeypatch.setattr(llm.httpx, "Client", lambda **kw: FakeClient(
        {"balance_infos": [{"currency": "CNY", "total_balance": "14.11",
                            "granted_balance": "0.46", "topped_up_balance": "13.65"}]}))
    d = client.quota("a")[0]
    assert d["supported"] and d["total"] == "14.11" and d["currency"] == "CNY"
