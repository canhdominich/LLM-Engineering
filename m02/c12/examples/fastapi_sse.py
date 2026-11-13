import time

import httpx

from _fake_stream import serve
from _sse_app import app


def measure(base, path):
    t0, first, events, raw = time.perf_counter(), None, 0, []
    with httpx.stream("POST", base + path, json={"message": "Đơn AT047838 đâu rồi?"}, timeout=10) as r:
        ctype = r.headers["content-type"]
        for line in r.iter_lines():
            raw.append(line)
            if line.startswith("event: delta"):
                events += 1
                first = first or time.perf_counter() - t0
    return ctype, events, first, time.perf_counter() - t0, raw


server, base = serve(app)
wire = {}
for path in ("/chat", "/chat-buffered"):
    ctype, events, first, total, wire[path] = measure(base, path)
    print(f"{path:<15} {ctype} | delta events {events:>2}, first text at {first:.1f} s, done at {total:.1f} s")
print("first lines on the wire of /chat:", wire["/chat"][:6])
print("last lines:", wire["/chat"][-3:-1])
server.should_exit = True
