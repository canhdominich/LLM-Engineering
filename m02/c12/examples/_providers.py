import os
from dataclasses import dataclass
from typing import Iterator, Protocol

import openai
from openai import OpenAI

from common.llm import FakeLLM


@dataclass
class Delta:
    text: str


@dataclass
class Done:
    input_tokens: int
    output_tokens: int
    model: str


class ProviderError(Exception):
    def __init__(self, code, retryable, message=""):
        super().__init__(f"{code}: {message}")
        self.code, self.retryable = code, retryable


class ChatProvider(Protocol):
    name: str

    def stream(self, messages: list[dict], max_tokens: int = 200) -> Iterator[Delta | Done]: ...


class ResponsesProvider:
    """OpenAI Responses API; works for OpenAI and for Ollama by changing base_url."""

    def __init__(self, name, model, base_url=None, api_key=None, extra=None, timeout=60.0):
        self.name, self.model, self.extra = name, model, extra or {}
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)

    def stream(self, messages, max_tokens=200):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        try:
            events = self.client.responses.create(model=self.model, instructions=system or None, input=rest,
                                                  max_output_tokens=max_tokens, stream=True, **self.extra)
            for ev in events:
                if ev.type == "response.output_text.delta":
                    yield Delta(ev.delta)
                elif ev.type == "response.completed":
                    u = ev.response.usage
                    yield Done(u.input_tokens, u.output_tokens, ev.response.model)
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise ProviderError(type(e).__name__, True, str(e)[:80]) from e  # lỗi tạm thời
        except openai.APIStatusError as e:
            raise ProviderError(type(e).__name__, False, str(e)[:80]) from e


class FakeProvider:
    """Wraps common.llm.FakeLLM and streams its scripted reply word by word."""

    def __init__(self, script):
        self.name, self.llm = "fake", FakeLLM(script)

    def stream(self, messages, max_tokens=200):
        reply = self.llm.chat(messages)
        for i, word in enumerate(reply.text.split(" ")):
            yield Delta(word if i == 0 else " " + word)
        yield Done(reply.usage.input_tokens, reply.usage.output_tokens, reply.model)


def make_provider(env=os.environ):
    kind = env.get("LLM_PROVIDER", "ollama")
    if kind == "ollama":
        return ResponsesProvider("ollama", env.get("LLM_MODEL", "qwen3:1.7b"), env.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                                 "ollama", extra={"reasoning": {"effort": "none"}, "temperature": 0})
    if kind == "openai":
        return ResponsesProvider("openai", env["LLM_MODEL"], None, env["OPENAI_API_KEY"])
    if kind == "fake":
        return FakeProvider([env.get("FAKE_REPLY", "Xin chào, mình là trợ lý thử nghiệm.")])
    raise ValueError(f"unknown LLM_PROVIDER {kind!r}")
