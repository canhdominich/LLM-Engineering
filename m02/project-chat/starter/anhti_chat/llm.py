import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Protocol

import openai
from openai import OpenAI


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass
class Completed:
    input_tokens: int
    output_tokens: int
    model: str
    seconds: float = 0.0
    ttft: float | None = None


class ProviderError(Exception):
    def __init__(self, code: str, retryable: bool, message: str = ""):
        super().__init__(f"{code}: {message}")
        self.code, self.retryable = code, retryable


Event = TextDelta | ToolCall | Completed


class LLMClient(Protocol):
    name: str
    model: str

    def stream(self, instructions: str, items: list[dict], tools: list[dict] | None = None,
               schema: dict | None = None, max_output_tokens: int = 300) -> Iterator[Event]: ...


def collect(events) -> tuple[str, list[ToolCall], Completed | None]:
    text, calls, done = "", [], None
    for ev in events:
        if isinstance(ev, TextDelta):
            text += ev.text
        elif isinstance(ev, ToolCall):
            calls.append(ev)
        elif isinstance(ev, Completed):
            done = ev
    return text, calls, done


class ResponsesClient:
    def __init__(self, name, model, base_url=None, api_key=None, extra=None, timeout=120.0, max_retries=2):
        self.name, self.model, self.extra = name, model, dict(extra or {})
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)

    def stream(self, instructions, items, tools=None, schema=None, max_output_tokens=300):
        kw = dict(model=self.model, instructions=instructions, input=items, max_output_tokens=max_output_tokens,
                  stream=True, **self.extra)
        if tools:
            kw["tools"] = tools
        if schema:
            kw["text"] = {"format": {"type": "json_schema", "name": schema["title"], "schema": schema, "strict": True}}
        t0, ttft = time.perf_counter(), None
        try:
            events = self.client.responses.create(**kw)
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise ProviderError(type(e).__name__, True, str(e)[:120]) from e
        except openai.APIStatusError as e:
            raise ProviderError(type(e).__name__, False, str(e)[:120]) from e
        try:
            for ev in events:
                if ev.type == "response.output_text.delta":
                    ttft = ttft if ttft is not None else time.perf_counter() - t0
                    yield TextDelta(ev.delta)
                elif ev.type == "response.output_item.done" and ev.item.type == "function_call":
                    yield ToolCall(ev.item.call_id, ev.item.name, ev.item.arguments)
                elif ev.type == "response.completed":
                    u = ev.response.usage
                    yield Completed(u.input_tokens, u.output_tokens, ev.response.model, time.perf_counter() - t0, ttft)
                elif ev.type in ("response.failed", "error"):
                    raise ProviderError("response_failed", True, str(ev)[:120])
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise ProviderError(type(e).__name__, True, str(e)[:120]) from e
        finally:
            events.close()  # đóng kết nối HTTP để provider ngừng sinh token


class FakeClient:
    """Adapter from common.llm.FakeLLM: each script item is a reply string or a dict with tool_calls."""

    def __init__(self, llm, delay: float = 0.0):
        self.llm, self.delay = llm, delay
        self.name, self.model = "fake", "fake"
        self.closed = 0
        self.words_sent = 0

    def stream(self, instructions, items, tools=None, schema=None, max_output_tokens=300):
        reply = self.llm.chat([{"role": "system", "content": instructions}] + items, tools=tools, schema=schema)
        try:
            for i, c in enumerate(reply.tool_calls):
                args = c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"])
                yield ToolCall(c.get("id", f"call_{i}"), c["name"], args)
            for i, word in enumerate(reply.text.split(" ") if reply.text else []):
                if self.delay:
                    time.sleep(self.delay)
                self.words_sent += 1
                yield TextDelta(word if i == 0 else " " + word)
            yield Completed(reply.usage.input_tokens, reply.usage.output_tokens, "fake")
        finally:
            self.closed += 1


class ReplayClient:
    """Record real replies once, replay them byte for byte in CI. A changed prompt is a cache miss."""

    def __init__(self, inner: LLMClient | None, cache_dir: Path, record: bool = False):
        self.inner, self.dir, self.record = inner, Path(cache_dir), record
        self.name = "replay"
        self.model = inner.model if inner else json.loads((self.dir / "MODEL").read_text())["model"]
        self.misses = 0

    def key(self, instructions, items, tools, schema, max_output_tokens):
        blob = json.dumps([self.model, instructions, items, tools, schema, max_output_tokens], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def stream(self, instructions, items, tools=None, schema=None, max_output_tokens=300):
        path = self.dir / (self.key(instructions, items, tools, schema, max_output_tokens) + ".json")
        if path.exists():
            for ev in json.loads(path.read_text(encoding="utf-8"))["events"]:
                kind = ev.pop("kind")
                yield {"delta": TextDelta, "call": ToolCall, "done": Completed}[kind](**ev)
            return
        self.misses += 1
        if not self.record or self.inner is None:
            raise ProviderError("replay_miss", False, f"no recording {path.name}; record with a real provider")
        saved = []
        for ev in self.inner.stream(instructions, items, tools, schema, max_output_tokens):
            kind = {TextDelta: "delta", ToolCall: "call", Completed: "done"}[type(ev)]
            saved.append({"kind": kind, **asdict(ev)})
            yield ev
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"events": saved}, ensure_ascii=False), encoding="utf-8")
        (self.dir / "MODEL").write_text(json.dumps({"model": self.model}))


def make_client(settings) -> LLMClient:
    if settings.provider == "ollama":
        return ResponsesClient("ollama", settings.model, settings.base_url, "ollama", timeout=settings.llm_timeout_s,
                               extra={"reasoning": {"effort": "none"}, "temperature": 0})
    if settings.provider == "openai":
        if not settings.api_key or settings.api_key == "ollama":
            raise ValueError("openai needs OPENAI_API_KEY and LLM_MODEL")
        return ResponsesClient("openai", settings.model, settings.base_url, settings.api_key, timeout=settings.llm_timeout_s)
    if settings.provider == "fake":
        from common.llm import FakeLLM
        return FakeClient(FakeLLM(['{"intent": "other", "order_id": null, "needs_human": false, "reason": "demo"}',
                                   "Xin chào, mình là trợ lý thử nghiệm của Giao Hàng Anh Tí."] * 500))
    raise ValueError(f"unknown LLM_PROVIDER {settings.provider!r}")
