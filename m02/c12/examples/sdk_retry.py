import time

import httpx
import openai
from openai import OpenAI

OK = {"id": "resp_1", "object": "response", "created_at": 0, "model": "qwen3:1.7b", "status": "completed",
      "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                  "content": [{"type": "output_text", "text": "Đơn AT047838 đang giao.", "annotations": []}]}],
      "usage": {"input_tokens": 40, "output_tokens": 9, "total_tokens": 49}}


def provider(script):
    attempts = []

    def handle(request):
        status = script[min(len(attempts), len(script) - 1)]
        attempts.append((round(time.perf_counter() - t0, 2), status))
        if status == 200:
            return httpx.Response(200, json=OK)
        headers = {"retry-after-ms": "300"} if status == 429 else {}  # provider bảo chờ 0,3 s
        return httpx.Response(status, json={"error": {"message": f"status {status}"}}, headers=headers)

    return handle, attempts


for name, script in [("429, 429, 200", [429, 429, 200]), ("500 always", [500]), ("400 bad request", [400])]:
    handle, attempts = provider(script)
    llm = OpenAI(base_url="http://provider.test/v1", api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    t0 = time.perf_counter()
    try:
        r = llm.responses.create(model="qwen3:1.7b", input="Đơn AT047838 đâu?")
        result = "ok: " + r.output_text
    except openai.APIStatusError as e:
        result = f"raised {type(e).__name__} (status {e.status_code})"
    starts = [a[0] - attempts[0][0] for a in attempts]
    print(f"{name:<16} attempts {len(attempts)}, started at {[round(x, 1) for x in starts]} s | {result}")
