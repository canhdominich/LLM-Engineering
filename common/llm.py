"""Provider-agnostic LLM client used by the lab code.

LLM_PROVIDER selects the backend:
- "ollama": local open-weight model through Ollama's OpenAI-compatible endpoint (default for the course)
- "openai": OpenAI API, needs OPENAI_API_KEY
- "fake": scripted FakeLLM for deterministic tests, no network
"""
import json
import os
import time
from dataclasses import dataclass, field

DEFAULTS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen3:1.7b", "api_key": "ollama"},
    "openai": {"base_url": None, "model": "gpt-6-luna", "api_key": None},
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Reply:
    text: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list = field(default_factory=list)
    model: str = ""
    seconds: float = 0.0


class FakeLLM:
    """Returns scripted replies in order; each item is a str or a dict with tool_calls."""

    def __init__(self, script, latency=0.0):
        self.script = list(script)
        self.calls = []
        self.latency = latency
        self.model = "fake"

    def chat(self, messages, tools=None, **kw):
        self.calls.append({"messages": messages, "tools": tools, **kw})
        if not self.script:
            raise RuntimeError("FakeLLM script exhausted")
        item = self.script.pop(0)
        time.sleep(self.latency)
        n_in = sum(len(str(m.get("content", ""))) for m in messages) // 4
        if isinstance(item, dict):
            return Reply(text=item.get("text", ""), tool_calls=item.get("tool_calls", []), usage=Usage(n_in, 20), model="fake")
        return Reply(text=item, usage=Usage(n_in, max(1, len(item) // 4)), model="fake")


class LLM:
    def __init__(self, provider=None, model=None):
        from openai import OpenAI
        self.provider = provider or os.environ.get("LLM_PROVIDER", "ollama")
        cfg = DEFAULTS[self.provider]
        self.model = model or os.environ.get("LLM_MODEL", cfg["model"])
        self.client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", cfg["base_url"]), api_key=os.environ.get("OPENAI_API_KEY", cfg["api_key"]))

    def chat(self, messages, tools=None, temperature=0.0, seed=7, response_format=None, max_tokens=None):
        t0 = time.perf_counter()
        kw = dict(model=self.model, messages=messages, temperature=temperature, seed=seed)
        if tools:
            kw["tools"] = tools
        if response_format:
            kw["response_format"] = response_format
        if max_tokens:
            kw["max_tokens"] = max_tokens
        r = self.client.chat.completions.create(**kw)
        msg = r.choices[0].message
        calls = [{"id": c.id, "name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")} for c in (msg.tool_calls or [])]
        u = Usage(r.usage.prompt_tokens, r.usage.completion_tokens) if r.usage else Usage()
        return Reply(text=msg.content or "", usage=u, tool_calls=calls, model=r.model, seconds=time.perf_counter() - t0)


def get_llm():
    if os.environ.get("LLM_PROVIDER") == "fake":
        raise RuntimeError("use FakeLLM(script) directly in tests")
    return LLM()
