import json

import psycopg
import pytest
from common.llm import FakeLLM

from anhti_chat.chat import ChatService
from anhti_chat.config import load_settings
from anhti_chat.llm import FakeClient
from anhti_chat.store import PgStore
from anhti_chat.tools import OrderBook, SnapshotEta, ToolRunner

AN, BINH, LAN, MINH, HA = "C01024", "C02210", "C03357", "C04481", "C05112"
H = {name: {"authorization": f"Bearer token-{name}"} for name in ("an", "binh", "lan", "minh", "ha")}


def triage(intent, order_id=None, needs_human=False, reason="test"):
    return json.dumps({"intent": intent, "order_id": order_id, "needs_human": needs_human, "reason": reason})


def call(name, order_id, i=1, **extra):
    return {"id": f"call_{i}", "name": name, "arguments": {"order_id": order_id, **extra}}


def sse_events(text):
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            out.append((name, json.loads("\n".join(data))))
    return out


class TrackedStore(PgStore):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.created = []

    def create_conversation(self, user_id):
        cid = super().create_conversation(user_id)
        self.created.append(cid)
        return cid


@pytest.fixture(scope="session")
def settings():
    s = load_settings(provider="fake", model="fake", pg_schema="m02_project_test")
    with psycopg.connect(s.pg_dsn) as conn:
        conn.execute("DROP SCHEMA IF EXISTS m02_project_test CASCADE")  # mỗi lần chạy test bắt đầu từ schema trống
    return s


@pytest.fixture
def store(settings):
    s = TrackedStore(settings.pg_dsn, settings.pg_schema).init()
    yield s
    for cid in s.created:
        s.delete_conversation(cid)


@pytest.fixture(scope="session")
def tools(settings):
    return ToolRunner(OrderBook(settings.orders_path), SnapshotEta(settings.eta_snapshot), settings.clock)


@pytest.fixture
def make(store, tools, settings):
    def factory(script, delay=0.0, **overrides):
        client = FakeClient(FakeLLM(script), delay=delay)
        s = settings if not overrides else type(settings)(**{**settings.__dict__, **overrides})
        return ChatService(client, store, tools, s), client
    return factory


def run(service, user, message, conversation_id=None):
    return list(service.handle(user, conversation_id, message))


def names(events):
    return [e.name for e in events]


def text_of(events):
    answer = "".join(e.data["text"] for e in events if e.name == "delta")
    for e in events:
        if e.name == "replace":
            answer = e.data["text"]
    return answer
