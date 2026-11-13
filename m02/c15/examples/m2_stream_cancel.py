import time

import _project  # noqa: F401
import httpx
from _server import LONG, last_assistant, serve, slow_service
from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from anhti_chat.api import create_app

AN = {"authorization": "Bearer token-an"}


def first_text(base, path):
    t0 = time.perf_counter()
    with httpx.stream("POST", base + path, headers=AN, json={"message": "Đơn AT047838 đâu rồi?"}, timeout=30) as r:
        for line in r.iter_lines():
            if "delta" in line or '"answer"' in line:
                first = time.perf_counter() - t0
        return first if path == "/chat/complete" else None, time.perf_counter() - t0


def ttft_sse(base):
    t0 = time.perf_counter()
    with httpx.stream("POST", base + "/chat", headers=AN, json={"message": "Đơn AT047838 đâu rồi?"}, timeout=30) as r:
        for line in r.iter_lines():
            if line == "event: delta":
                return time.perf_counter() - t0


def stop_after(base, n):
    seen = 0
    with httpx.stream("POST", base + "/chat", headers=AN, json={"message": "Kể chi tiết đi"}, timeout=30) as r:
        for line in r.iter_lines():
            seen += line == "event: delta"
            if seen == n:
                return  # đóng kết nối giống người dùng bấm Dừng


service, llm = slow_service(rounds=3)
server, base = serve(create_app(service))
print(f"SSE /chat: first text after {ttft_sse(base):.2f} s")
_, total = first_text(base, "/chat/complete")
print(f"JSON /chat/complete: whole answer (first text) after {total:.2f} s")
start = llm.words_sent + 8  # 8 từ đầu là JSON của lời gọi phân loại
stop_after(base, 5)
time.sleep(1.5)
row = last_assistant(service.store)
print(f"with watcher: client read 5 words, model sent {llm.words_sent - start} of {len(LONG.split())}, "
      f"provider streams closed = {llm.closed} of 6, saved = {row['status']} {row['content']!r}")
server.should_exit = True


class ChatIn(BaseModel):
    message: str


naive_service, naive_llm = slow_service()
naive = FastAPI()


@naive.post("/chat", response_class=EventSourceResponse)
def naive_chat(body: ChatIn):
    for ev in naive_service.handle("C01024", None, body.message):  # không có task theo dõi ngắt kết nối
        yield ServerSentEvent(data=ev.data, event=ev.name)


server, base = serve(naive)
stop_after(base, 5)
time.sleep(1.5)
row = last_assistant(naive_service.store)
print(f"naive endpoint: provider streams closed = {naive_llm.closed} of 2, saved assistant row = {row}")
server.should_exit = True
