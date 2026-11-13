import socket
import threading
import time

import uvicorn
from common.llm import FakeLLM
from conftest import triage

from anhti_chat.chat import ChatService
from anhti_chat.config import load_settings
from anhti_chat.llm import FakeClient
from anhti_chat.store import MemoryStore
from anhti_chat.tools import OrderBook, SnapshotEta, ToolRunner

LONG = " ".join(["Đơn AT047838 đang được giao ở Cho Moi, shipper sẽ gọi trước khi tới."] * 5)


def slow_service(answer=LONG, prefill=1.0, per_word=0.12, rounds=1):
    """FakeLLM that waits `prefill` seconds then sends one word every `per_word` seconds (about 8 tokens/s, like qwen3 on CPU)."""
    settings = load_settings(provider="fake")
    llm = FakeClient(FakeLLM([triage("other"), answer] * rounds, latency=prefill / 2), delay=per_word)
    tools = ToolRunner(OrderBook(settings.orders_path), SnapshotEta(settings.eta_snapshot), settings.clock)
    return ChatService(llm, MemoryStore(), tools, settings), llm


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


def last_assistant(store):
    cid = list(store.msgs)[-1]
    rows = [m for m in store.msgs[cid] if m["role"] == "assistant"]
    return rows[-1] if rows else None
