import json

import pytest
from common.llm import FakeLLM
from fastapi.testclient import TestClient

import chat_api as ca


def events(text):
    out = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        out.append((name, json.loads("\n".join(data))))
    return out


@pytest.fixture(scope="module")
def store():
    return ca.PgStore().init()


@pytest.fixture
def created(store):
    ids = []
    yield ids
    for cid in ids:
        store.delete(cid)


def post(client, created, **body):
    r = client.post("/chat", json=body)
    evs = events(r.text) if r.status_code == 200 else []
    if evs and evs[0][0] == "meta":
        created.append(evs[0][1]["conversation_id"])
    return r, evs


class Broken:
    name = "broken"

    def stream(self, messages, max_tokens=300):
        yield ca.Delta("Đơn")
        raise ca.ProviderError("RateLimitError", True, "slow down")


def test_trim_history_keeps_whole_last_turns():
    msgs = [{"role": r, "content": str(i)} for i, r in enumerate(["user", "assistant"] * 5 + ["user"])]
    kept = ca.trim_history(msgs, 3)
    assert [m["content"] for m in kept] == ["6", "7", "8", "9", "10"]
    assert kept[0]["role"] == "user"
    assert ca.trim_history(msgs[:3], 3) == msgs[:3]


def test_fake_provider_streams_words_then_done():
    p = ca.FakeLLMProvider(FakeLLM(["Đơn AT047838 đang giao."]))
    items = list(p.stream([{"role": "user", "content": "hi"}]))
    assert [i.text for i in items[:-1]] == ["Đơn", " AT047838", " đang", " giao."]
    assert isinstance(items[-1], ca.Done) and items[-1].model == "fake" and items[-1].output_tokens > 0


def test_chat_streams_meta_deltas_done(store, created):
    client = TestClient(ca.create_app(ca.FakeLLMProvider(FakeLLM(["Đơn AT047838 đang ở Cho Moi."])), store))
    r, evs = post(client, created, message="Đơn AT047838 đâu rồi?")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    names = [e for e, _ in evs]
    assert names[0] == "meta" and names[-1] == "done" and set(names[1:-1]) == {"delta"} and len(names) == 8
    assert "".join(d["text"] for e, d in evs if e == "delta") == "Đơn AT047838 đang ở Cho Moi."
    assert evs[0][1]["provider"] == "fake" and evs[-1][1]["model"] == "fake"


def test_messages_are_saved_with_usage(store, created):
    client = TestClient(ca.create_app(ca.FakeLLMProvider(FakeLLM(["Chào Lan."])), store))
    _, evs = post(client, created, message="Chào bạn, tôi là Lan.")
    cid = evs[0][1]["conversation_id"]
    rows = client.get(f"/conversations/{cid}/messages").json()
    assert [(m["seq"], m["role"], m["content"]) for m in rows] == [(1, "user", "Chào bạn, tôi là Lan."), (2, "assistant", "Chào Lan.")]
    assert rows[1]["output_tokens"] == evs[-1][1]["output_tokens"] and rows[0]["output_tokens"] is None


def test_second_turn_sends_system_and_history(store, created):
    llm = FakeLLM(["Đã ghi nhận đơn AT047838.", "Đơn giao tới Ben Song."])
    client = TestClient(ca.create_app(ca.FakeLLMProvider(llm), store, system_prompt="SYS"))
    _, evs = post(client, created, message="Đơn AT047838 giao tới Ben Song.")
    cid = evs[0][1]["conversation_id"]
    post(client, created, conversation_id=cid, message="Đơn giao tới đâu?")
    sent = llm.calls[1]["messages"]
    assert sent[0] == {"role": "system", "content": "SYS"}
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "user"]
    assert sent[2]["content"] == "Đã ghi nhận đơn AT047838." and sent[3]["content"] == "Đơn giao tới đâu?"
    assert len(created) == 2 and created[0] == created[1]


def test_history_is_trimmed_before_calling_model(store, created):
    llm = FakeLLM([f"reply {i}" for i in range(4)])
    client = TestClient(ca.create_app(ca.FakeLLMProvider(llm), store, max_turns=2))
    _, evs = post(client, created, message="turn 0")
    cid = evs[0][1]["conversation_id"]
    for i in range(1, 4):
        post(client, created, conversation_id=cid, message=f"turn {i}")
    sent = llm.calls[3]["messages"]
    assert [m["content"] for m in sent[1:]] == ["turn 2", "reply 2", "turn 3"]
    assert len(store.history(cid)) == 8


def test_history_survives_a_new_store_object(store, created):
    client = TestClient(ca.create_app(ca.FakeLLMProvider(FakeLLM(["ok"])), store))
    _, evs = post(client, created, message="ghi nhớ giúp tôi")
    cid = evs[0][1]["conversation_id"]
    again = ca.PgStore()
    assert again.exists(cid) and [m["role"] for m in again.history(cid)] == ["user", "assistant"]


def test_unknown_conversation_and_bad_body(store, created):
    client = TestClient(ca.create_app(ca.FakeLLMProvider(FakeLLM(["never"])), store))
    r, _ = post(client, created, conversation_id="does-not-exist", message="hi")
    assert r.status_code == 404
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.get("/conversations/does-not-exist/messages").status_code == 404


def test_provider_error_becomes_error_event(store, created):
    client = TestClient(ca.create_app(Broken(), store))
    r, evs = post(client, created, message="Đơn AT047838 đâu?")
    assert r.status_code == 200
    assert [e for e, _ in evs] == ["meta", "delta", "error"]
    assert evs[-1][1] == {"code": "RateLimitError", "retryable": True}
    cid = evs[0][1]["conversation_id"]
    assert [m["role"] for m in store.history(cid)] == ["user"]


def test_make_provider_from_env():
    assert ca.make_provider({"LLM_PROVIDER": "fake"}).name == "fake"
    p = ca.make_provider({"LLM_PROVIDER": "ollama"})
    assert p.name == "ollama" and p.model == "qwen3:1.7b" and str(p.client.base_url).startswith("http://localhost:11434/v1")
    assert ca.make_provider({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "LLM_MODEL": "m"}).name == "openai"
    with pytest.raises(ValueError):
        ca.make_provider({"LLM_PROVIDER": "openai"})
    with pytest.raises(ValueError):
        ca.make_provider({"LLM_PROVIDER": "mystery"})
