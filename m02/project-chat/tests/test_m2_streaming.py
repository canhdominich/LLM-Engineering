import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from conftest import AN, H, names, sse_events, triage
from fastapi.testclient import TestClient

from anhti_chat.api import STATIC, create_app
from anhti_chat.llm import Completed, ProviderError, TextDelta

FRONTEND = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
ANSWER = "Đơn AT047838 đang được giao ở Cho Moi, shipper sắp tới nơi rồi."


class Broken:
    name, model = "broken", "broken"

    def stream(self, instructions, items, tools=None, schema=None, max_output_tokens=300):
        if schema:
            yield TextDelta(triage("order_status", "AT047838"))
            yield Completed(10, 5, "broken")
            return
        yield TextDelta("Đơn AT047838")
        raise ProviderError("RateLimitError", True, "slow down")


def serve(app):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.02)
    return server, f"http://127.0.0.1:{port}"


def test_sse_stream_sends_meta_triage_deltas_done(make):
    service, _ = make([triage("other"), ANSWER])
    r = TestClient(create_app(service)).post("/chat", headers=H["an"], json={"message": "Đơn AT047838 đâu rồi?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = sse_events(r.text)
    kinds = [e for e, _ in evs]
    assert kinds[:2] == ["meta", "triage"] and kinds[-1] == "done" and set(kinds[2:-1]) == {"delta"}
    assert "".join(d["text"] for e, d in evs if e == "delta") == ANSWER
    assert kinds.count("delta") == len(ANSWER.split(" "))


def test_provider_error_becomes_an_error_event_and_partial_text_is_saved(store, tools, settings):
    from anhti_chat.chat import ChatService
    service = ChatService(Broken(), store, tools, settings)
    events = list(service.handle(AN, None, "Đơn AT047838 đâu?"))
    assert names(events)[-1] == "error" and events[-1].data == {"code": "RateLimitError", "retryable": True}
    last = store.messages(events[0].data["conversation_id"])[-1]
    assert (last["role"], last["content"], last["status"]) == ("assistant", "Đơn AT047838", "error")


def test_cancel_flag_stops_the_model_stream_and_saves_partial_answer(make, store):
    service, client = make([triage("other"), " ".join(["chữ"] * 50)])
    cancel = threading.Event()
    gen = service.handle(AN, None, "Kể chuyện dài đi", cancel)
    seen, after_triage = [], 0
    for ev in gen:
        seen.append(ev)
        if ev.name == "triage":
            after_triage = client.words_sent
        if names(seen).count("delta") == 3:
            cancel.set()
    assert names(seen)[-1] == "done" and seen[-1].data["stopped"] == "cancelled"
    last = store.messages(seen[0].data["conversation_id"])[-1]
    assert last["status"] == "cancelled" and last["content"] == "chữ chữ chữ"
    assert client.closed == 2 and client.words_sent - after_triage == 4


def test_closing_the_generator_saves_cancelled_answer(make, store):
    service, client = make([triage("other"), "một hai ba bốn năm sáu"])
    gen = service.handle(AN, None, "Đếm đi")
    first = next(gen)
    while next(gen).name != "delta":
        pass
    gen.close()
    last = store.messages(first.data["conversation_id"])[-1]
    assert (last["content"], last["status"]) == ("một", "cancelled")
    assert client.closed == 2


def test_client_disconnect_stops_generation_on_a_real_server(make, store):
    service, client = make([triage("other"), " ".join(["chữ"] * 60)], delay=0.05)
    server, base = serve(create_app(service))
    try:
        deltas = 0
        with httpx.stream("POST", base + "/chat", headers=H["an"], json={"message": "Kể chuyện dài đi"}, timeout=10) as r:
            for line in r.iter_lines():
                if line.startswith("event: delta"):
                    deltas += 1
                if deltas == 3:
                    break
        deadline = time.time() + 5
        while time.time() < deadline and not (store.created and store.messages(store.created[-1])[-1]["role"] == "assistant"):
            time.sleep(0.05)
        last = store.messages(store.created[-1])[-1]
        assert last["status"] == "cancelled"
        assert client.words_sent < 25
    finally:
        server.should_exit = True


def test_frontend_is_served_and_holds_no_secret(make):
    service, _ = make([])
    api = TestClient(create_app(service))
    page = api.get("/")
    assert page.status_code == 200 and 'src="app.js"' in page.text
    for name in ("app.js", "sse.js"):
        r = api.get("/" + name)
        assert r.status_code == 200 and "javascript" in r.headers["content-type"]
        assert "OPENAI_API_KEY" not in r.text and "sk-" not in r.text and ".innerHTML" not in r.text
    assert "AbortController" in api.get("/app.js").text and "controller?.abort()" in api.get("/app.js").text


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_typescript_parser_reads_what_the_server_sends(make):
    service, _ = make([triage("other"), ANSWER])
    body = TestClient(create_app(service)).post("/chat", headers=H["an"], json={"message": "hi"}).text
    body = ": ping\n\n" + body
    script = ("import { parseSse } from '" + (STATIC / "sse.js").as_uri() + "';"
              "let s=''; process.stdin.on('data', d => s += d); process.stdin.on('end', () => {"
              "const half = s.slice(0, 200); const a = parseSse(half); const b = parseSse(a.rest + s.slice(200));"
              "console.log(JSON.stringify(a.events.concat(b.events)))});")
    p = subprocess.run([NODE, "--no-warnings", "--input-type=module", "-e", script], input=body, capture_output=True, text=True, timeout=20)
    assert p.returncode == 0, p.stderr[-500:]
    got = [(e["event"], e["data"]) for e in json.loads(p.stdout)]
    assert got == sse_events(body)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_static_js_is_built_from_the_typescript_source():
    impl = STATIC.parents[1]
    script = ("import { stripTypeScriptTypes } from 'node:module'; import { readFileSync } from 'node:fs';"
              "const [src, out] = process.argv.slice(1);"
              "process.exit(stripTypeScriptTypes(readFileSync(src, 'utf8')) === readFileSync(out, 'utf8') ? 0 : 1);")
    for name in ("sse", "app"):
        p = subprocess.run([NODE, "--no-warnings", "--input-type=module", "-e", script, str(impl / "frontend" / "src" / f"{name}.ts"),
                            str(STATIC / f"{name}.js")], capture_output=True, text=True, timeout=20)
        assert p.returncode == 0, f"static/{name}.js is stale: run node frontend/build.mjs"
