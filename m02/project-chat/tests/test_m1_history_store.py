import psycopg
from conftest import AN, BINH, H, run, triage
from fastapi.testclient import TestClient

from anhti_chat.api import create_app
from anhti_chat.history import build_context, estimate_tokens


def msgs(*pairs):
    return [{"seq": i + 1, "role": r, "content": c, "status": "complete"} for i, (r, c) in enumerate(pairs)]


def test_schema_has_the_five_tables(store, settings):
    with psycopg.connect(settings.pg_dsn) as conn:
        rows = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (settings.pg_schema,)).fetchall()
    assert {"conversations", "messages", "usage_events", "tool_calls", "handoffs"} <= {r[0] for r in rows}


def test_conversation_is_visible_only_to_its_owner(store):
    cid = store.create_conversation(AN)
    assert store.get_conversation(cid, AN)["user_id"] == AN
    assert store.get_conversation(cid, BINH) is None
    assert cid in [c["id"] for c in store.list_conversations(AN)]
    assert cid not in [c["id"] for c in store.list_conversations(BINH)]


def test_messages_keep_order_and_status(store):
    cid = store.create_conversation(AN)
    assert [store.add_message(cid, "user", "Đơn AT047838 đâu?"), store.add_message(cid, "assistant", "Đang giao.", "replaced")] == [1, 2]
    got = store.messages(cid)
    assert [(m["seq"], m["role"], m["status"]) for m in got] == [(1, "user", "complete"), (2, "assistant", "replaced")]


def test_every_llm_call_writes_a_usage_event(make, store):
    service, _ = make([triage("other"), "Chào An, mình giúp gì được cho bạn?"])
    events = run(service, AN, "Chào bạn")
    cid = events[0].data["conversation_id"]
    usage = store.usage(cid)
    assert [u["kind"] for u in usage] == ["triage", "chat"]
    assert all(u["input_tokens"] > 0 and u["output_tokens"] > 0 and u["model"] == "fake" for u in usage)
    assert events[-1].name == "done" and events[-1].data["calls"] == 2
    assert [m["role"] for m in store.messages(cid)] == ["user", "assistant"]


def test_build_context_keeps_whole_recent_turns_within_budget():
    history = msgs(("user", "a" * 300), ("assistant", "b" * 300), ("user", "c" * 300), ("assistant", "d" * 300), ("user", "e" * 30))
    items, older = build_context(history, 0, budget=250)
    assert [m["content"][0] for m in items] == ["c", "d", "e"]
    assert [m["seq"] for m in older] == [1, 2]
    assert sum(estimate_tokens(m["content"]) for m in items) <= 250


def test_build_context_always_keeps_the_last_user_message():
    items, older = build_context(msgs(("user", "x" * 3000)), 0, budget=50)
    assert len(items) == 1 and older == []


def test_build_context_skips_messages_already_in_the_summary():
    history = msgs(("user", "old"), ("assistant", "old answer"), ("user", "new"))
    items, older = build_context(history, summary_upto=2, budget=1000)
    assert items == [{"role": "user", "content": "new"}] and older == []


def test_long_history_is_summarized_and_summary_goes_to_the_model(make, store):
    service, client = make(["Khách An hỏi về đơn AT051122 giao tới Doi Thong.", triage("eta", "AT051122"), "Bạn chờ chút nhé."],
                           history_budget_tokens=60)
    cid = store.create_conversation(AN)
    for i in range(4):
        store.add_message(cid, "user", f"Câu hỏi số {i} về đơn AT051122 giao tới Doi Thong, mình cần biết thêm chi tiết.")
        store.add_message(cid, "assistant", f"Trả lời số {i}: mình đã ghi nhận thông tin của bạn.")
    run(service, AN, "Vậy bao giờ tới?", cid)
    conv = store.get_conversation(cid, AN)
    assert conv["summary"].startswith("Khách An") and conv["summary_upto"] >= 2
    triage_call = client.llm.calls[1]["messages"]
    assert "Conversation summary so far: Khách An" in triage_call[0]["content"]
    assert len(triage_call) - 1 < 9
    assert [u["kind"] for u in store.usage(cid)][0] == "summary"


def test_api_requires_a_valid_token(make):
    service, _ = make([])
    client = TestClient(create_app(service))
    assert client.post("/chat/complete", json={"message": "hi"}).status_code == 401
    assert client.get("/conversations", headers={"authorization": "Bearer token-nobody"}).status_code == 401


def test_api_does_not_reveal_other_users_conversations(make, store):
    service, _ = make([triage("other"), "Chào An."])
    client = TestClient(create_app(service))
    r = client.post("/chat/complete", headers=H["an"], json={"message": "Chào bạn"})
    cid = r.json()["conversation_id"]
    assert client.get(f"/conversations/{cid}/messages", headers=H["an"]).status_code == 200
    assert client.get(f"/conversations/{cid}/messages", headers=H["binh"]).status_code == 404
    assert client.post("/chat/complete", headers=H["binh"], json={"conversation_id": cid, "message": "hi"}).status_code == 404
    assert client.post("/chat", headers=H["binh"], json={"conversation_id": cid, "message": "hi"}).status_code == 404


def test_chat_complete_returns_answer_and_keeps_multi_turn_history(make, store):
    service, client = make([triage("other"), "Chào An.", triage("other"), "Bạn tên là An."])
    api = TestClient(create_app(service))
    first = api.post("/chat/complete", headers=H["an"], json={"message": "Chào bạn, mình là An"}).json()
    second = api.post("/chat/complete", headers=H["an"], json={"conversation_id": first["conversation_id"], "message": "Mình tên gì?"})
    assert second.status_code == 200 and second.json()["answer"] == "Bạn tên là An."
    sent = client.llm.calls[3]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert second.json()["events"][-1]["event"] == "done"
