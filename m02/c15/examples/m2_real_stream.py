import json
import time

import _project  # noqa: F401
import httpx
from _server import serve

from anhti_chat.api import build_service, create_app
from anhti_chat.config import load_settings
from anhti_chat.store import MemoryStore

settings = load_settings(provider="ollama")
service = build_service(settings, store=MemoryStore())
server, base = serve(create_app(service))
t0, answer = time.perf_counter(), ""
with httpx.stream("POST", base + "/chat", headers={"authorization": "Bearer token-an"},
                  json={"message": "Đơn AT047838 bao lâu nữa tới?"}, timeout=600) as r:
    event = None
    for line in r.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
            at = f"{time.perf_counter() - t0:6.1f} s"
            if event == "delta":
                if not answer:
                    print(at, "first delta (TTFT)")
                answer += data["text"]
            elif event == "triage":
                print(at, "triage", data["intent"], data["order_id"])
            elif event == "tool":
                print(at, "tool", data["name"], data["result"])
            elif event == "done":
                print(at, "done", {k: data[k] for k in ("stopped", "steps", "calls", "input_tokens", "output_tokens")})
print("answer:", answer)
print("usage rows:", [(u["kind"], u["input_tokens"], u["output_tokens"], u["latency_ms"]) for u in service.store.usage(list(service.store.msgs)[-1])])
server.should_exit = True
