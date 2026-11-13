import os
import uuid
from dataclasses import dataclass
from typing import Annotated, Iterator, Protocol

import openai
import psycopg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai import OpenAI
from pydantic import BaseModel, Field

from common.llm import FakeLLM

DSN = os.environ.get("ANHTI_PG_DSN", "postgresql://anhti:anhti@localhost:55432/anhti")
SCHEMA = os.environ.get("ANHTI_PG_SCHEMA", "m02_c12")
SYSTEM_PROMPT = "Bạn là trợ lý chat của Giao Hang Anh Ti ở Pho Rom Ra. Trả lời ngắn gọn bằng tiếng Việt."


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

    def stream(self, messages: list[dict], max_tokens: int = 300) -> Iterator[Delta | Done]: ...


class FakeLLMProvider:
    def __init__(self, llm: FakeLLM):
        self.name, self.llm = "fake", llm

    def stream(self, messages, max_tokens=300):
        reply = self.llm.chat(messages, max_tokens=max_tokens)
        for i, word in enumerate(reply.text.split(" ")):
            yield Delta(word if i == 0 else " " + word)
        yield Done(reply.usage.input_tokens, reply.usage.output_tokens, reply.model)


class ResponsesProvider:
    def __init__(self, name, model, base_url=None, api_key=None, extra=None, timeout=60.0, max_retries=2):
        self.name, self.model, self.extra = name, model, dict(extra or {})
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)

    def stream(self, messages, max_tokens=300):
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
            raise ProviderError(type(e).__name__, True, str(e)[:120]) from e
        except openai.APIStatusError as e:
            raise ProviderError(type(e).__name__, False, str(e)[:120]) from e


def make_provider(env=None):
    env = os.environ if env is None else env
    kind = env.get("LLM_PROVIDER", "ollama")
    if kind == "ollama":
        return ResponsesProvider("ollama", env.get("LLM_MODEL", "qwen3:1.7b"), env.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                                 "ollama", extra={"reasoning": {"effort": "none"}, "temperature": 0})
    if kind == "openai":
        if not env.get("OPENAI_API_KEY") or not env.get("LLM_MODEL"):
            raise ValueError("openai needs OPENAI_API_KEY and LLM_MODEL")
        return ResponsesProvider("openai", env["LLM_MODEL"], env.get("LLM_BASE_URL"), env["OPENAI_API_KEY"])
    if kind == "fake":
        return FakeLLMProvider(FakeLLM([env.get("FAKE_REPLY", "Xin chào, mình là trợ lý thử nghiệm.")] * 100))
    raise ValueError(f"unknown LLM_PROVIDER {kind!r}")


class PgStore:
    def __init__(self, dsn=DSN, schema=SCHEMA):
        self.dsn, self.s = dsn, schema

    def init(self):
        with psycopg.connect(self.dsn) as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.s}")
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {self.s}.chat_conversations (
                id text PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {self.s}.chat_messages (
                conversation_id text NOT NULL REFERENCES {self.s}.chat_conversations(id) ON DELETE CASCADE,
                seq integer NOT NULL,
                role text NOT NULL CHECK (role IN ('user', 'assistant')),
                content text NOT NULL,
                input_tokens integer,
                output_tokens integer,
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (conversation_id, seq))""")
        return self

    def new_conversation(self):
        cid = uuid.uuid4().hex
        with psycopg.connect(self.dsn) as conn:
            conn.execute(f"INSERT INTO {self.s}.chat_conversations (id) VALUES (%s)", (cid,))
        return cid

    def exists(self, cid):
        with psycopg.connect(self.dsn) as conn:
            return conn.execute(f"SELECT 1 FROM {self.s}.chat_conversations WHERE id = %s", (cid,)).fetchone() is not None

    def add(self, cid, role, content, input_tokens=None, output_tokens=None):
        with psycopg.connect(self.dsn) as conn:
            conn.execute(f"SELECT id FROM {self.s}.chat_conversations WHERE id = %s FOR UPDATE", (cid,))
            seq = conn.execute(f"SELECT coalesce(max(seq), 0) + 1 FROM {self.s}.chat_messages WHERE conversation_id = %s",
                               (cid,)).fetchone()[0]
            conn.execute(f"""INSERT INTO {self.s}.chat_messages (conversation_id, seq, role, content, input_tokens, output_tokens)
                VALUES (%s, %s, %s, %s, %s, %s)""", (cid, seq, role, content, input_tokens, output_tokens))
        return seq

    def history(self, cid):
        with psycopg.connect(self.dsn) as conn:
            rows = conn.execute(f"""SELECT seq, role, content, input_tokens, output_tokens FROM {self.s}.chat_messages
                WHERE conversation_id = %s ORDER BY seq""", (cid,)).fetchall()
        return [{"seq": r[0], "role": r[1], "content": r[2], "input_tokens": r[3], "output_tokens": r[4]} for r in rows]

    def delete(self, cid):
        with psycopg.connect(self.dsn) as conn:
            conn.execute(f"DELETE FROM {self.s}.chat_conversations WHERE id = %s", (cid,))


def trim_history(messages, max_turns):
    starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(starts) <= max_turns:
        return list(messages)
    return list(messages[starts[-max_turns]:])


class ChatIn(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


def create_app(provider, store, system_prompt=SYSTEM_PROMPT, max_turns=8, max_tokens=300):
    app = FastAPI(title="Giao Hang Anh Ti chat")

    def checked(body: ChatIn) -> ChatIn:
        if body.conversation_id is not None and not store.exists(body.conversation_id):
            raise HTTPException(404, "conversation_not_found")
        return body

    @app.post("/chat", response_class=EventSourceResponse)
    def chat(body: Annotated[ChatIn, Depends(checked)]):
        cid = body.conversation_id or store.new_conversation()
        store.add(cid, "user", body.message)
        past = [{"role": m["role"], "content": m["content"]} for m in store.history(cid)]
        messages = [{"role": "system", "content": system_prompt}] + trim_history(past, max_turns)
        yield ServerSentEvent(data={"conversation_id": cid, "provider": provider.name}, event="meta")
        text = ""
        try:
            for item in provider.stream(messages, max_tokens=max_tokens):
                if isinstance(item, Delta):
                    text += item.text
                    yield ServerSentEvent(data={"text": item.text}, event="delta")
                elif isinstance(item, Done):
                    store.add(cid, "assistant", text, item.input_tokens, item.output_tokens)
                    yield ServerSentEvent(data={"input_tokens": item.input_tokens, "output_tokens": item.output_tokens,
                                                "model": item.model}, event="done")
        except ProviderError as e:
            yield ServerSentEvent(data={"code": e.code, "retryable": e.retryable}, event="error")

    @app.get("/conversations/{cid}/messages")
    def messages(cid: str):
        if not store.exists(cid):
            raise HTTPException(404, "conversation_not_found")
        return store.history(cid)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(make_provider(), PgStore().init()), host="127.0.0.1", port=8000)
