import _project  # noqa: F401
import psycopg
from common.llm import FakeLLM
from conftest import triage
from fastapi.testclient import TestClient

from anhti_chat.api import create_app
from anhti_chat.chat import ChatService
from anhti_chat.config import load_settings
from anhti_chat.llm import FakeClient
from anhti_chat.store import PgStore
from anhti_chat.tools import OrderBook, SnapshotEta, ToolRunner

settings = load_settings(provider="fake")
store = PgStore(settings.pg_dsn, settings.pg_schema).init()
llm = FakeClient(FakeLLM([triage("other"), "Chào An, mình là trợ lý của Giao Hàng Anh Tí.",
                          triage("other"), "Bạn vừa giới thiệu tên là An."]))
tools = ToolRunner(OrderBook(settings.orders_path), SnapshotEta(settings.eta_snapshot), settings.clock)
api = TestClient(create_app(ChatService(llm, store, tools, settings)))
AN, BINH = {"authorization": "Bearer token-an"}, {"authorization": "Bearer token-binh"}

r1 = api.post("/chat/complete", headers=AN, json={"message": "Chào bạn, mình là An"}).json()
cid = r1["conversation_id"]
r2 = api.post("/chat/complete", headers=AN, json={"conversation_id": cid, "message": "Mình tên gì nhỉ?"}).json()
print("turn 2 answer:", r2["answer"])
print("model saw on turn 2:", [m["role"] for m in llm.llm.calls[3]["messages"]])
print("no token:", api.get(f"/conversations/{cid}/messages").status_code)
print("Binh reads An's conversation:", api.get(f"/conversations/{cid}/messages", headers=BINH).status_code)
print("Binh posts into it:", api.post("/chat/complete", headers=BINH, json={"conversation_id": cid, "message": "hi"}).status_code)
with psycopg.connect(settings.pg_dsn) as conn:
    for table, cols in [("messages", "seq, role, status, left(content, 40)"), ("usage_events", "kind, model, input_tokens, output_tokens")]:
        print(f"{settings.pg_schema}.{table}:")
        for row in conn.execute(f"SELECT {cols} FROM {settings.pg_schema}.{table} WHERE conversation_id = %s", (cid,)):
            print("  ", row)
store.delete_conversation(cid)
