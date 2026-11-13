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
        # TODO 1: gọi self.llm.chat(messages, max_tokens=max_tokens) đúng một lần, yield Delta cho từng từ
        # (từ thứ hai trở đi có dấu cách ở đầu), cuối cùng yield Done(input_tokens, output_tokens, model) lấy từ reply
        raise NotImplementedError


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
    # TODO 2: đọc env["LLM_PROVIDER"] (mặc định "ollama") và trả về:
    # "ollama": ResponsesProvider("ollama", LLM_MODEL hoặc "qwen3:1.7b", LLM_BASE_URL hoặc "http://localhost:11434/v1", "ollama",
    #           extra={"reasoning": {"effort": "none"}, "temperature": 0})
    # "openai": cần OPENAI_API_KEY và LLM_MODEL, thiếu thì ValueError; "fake": FakeLLMProvider với FAKE_REPLY
    # giá trị khác: ValueError
    raise NotImplementedError


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
        # TODO 3: trong một transaction, khóa dòng hội thoại (SELECT ... FOR UPDATE), tính seq = max(seq) + 1
        # (bắt đầu từ 1), INSERT vào chat_messages bằng tham số %s, trả về seq
        raise NotImplementedError

    def history(self, cid):
        # TODO 4: trả về list dict có seq, role, content, input_tokens, output_tokens, sắp theo seq tăng dần
        raise NotImplementedError

    def delete(self, cid):
        with psycopg.connect(self.dsn) as conn:
            conn.execute(f"DELETE FROM {self.s}.chat_conversations WHERE id = %s", (cid,))


def trim_history(messages, max_turns):
    # TODO 5: giữ max_turns lượt cuối, mỗi lượt bắt đầu bằng một tin nhắn user; không cắt giữa một lượt
    raise NotImplementedError


class ChatIn(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


def create_app(provider, store, system_prompt=SYSTEM_PROMPT, max_turns=8, max_tokens=300):
    app = FastAPI(title="Giao Hang Anh Ti chat")

    def checked(body: ChatIn) -> ChatIn:
        # TODO 6: nếu body.conversation_id có giá trị mà store không có, raise HTTPException(404, "conversation_not_found")
        return body

    @app.post("/chat", response_class=EventSourceResponse)
    def chat(body: Annotated[ChatIn, Depends(checked)]):
        # TODO 7: tạo hội thoại nếu chưa có, lưu tin nhắn user, dựng messages = [system] + trim_history(lịch sử, max_turns),
        # yield sự kiện "meta" {conversation_id, provider}, mỗi Delta thành sự kiện "delta" {text},
        # khi có Done thì lưu câu trả lời kèm usage rồi yield "done" {input_tokens, output_tokens, model},
        # ProviderError thành sự kiện "error" {code, retryable} và không lưu câu trả lời
        raise NotImplementedError
        yield

    @app.get("/conversations/{cid}/messages")
    def messages(cid: str):
        if not store.exists(cid):
            raise HTTPException(404, "conversation_not_found")
        return store.history(cid)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(make_provider(), PgStore().init()), host="127.0.0.1", port=8000)
