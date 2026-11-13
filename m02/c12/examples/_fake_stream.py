import asyncio
import socket
import threading
import time

import uvicorn


class FakeStreamLLM:
    """Scripted model: waits `prefill` seconds, then yields one word every `per_token` seconds."""

    def __init__(self, text, prefill=0.6, per_token=0.08):
        self.words = text.split(" ")
        self.prefill, self.per_token = prefill, per_token

    async def stream(self, messages):
        await asyncio.sleep(self.prefill)
        for i, w in enumerate(self.words):
            await asyncio.sleep(self.per_token)
            yield w if i == 0 else " " + w


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
