import time
from collections import Counter

from _setup import INSTRUCTIONS, MODEL, Timer, client

llm = client()
events, deltas, first = Counter(), [], None
with Timer() as t:
    stream = llm.responses.create(model=MODEL, instructions=INSTRUCTIONS, input="Đơn hàng giao trễ thì tôi được hỗ trợ gì?",
                                  max_output_tokens=120, temperature=0, reasoning={"effort": "none"}, stream=True)
    for ev in stream:
        events[ev.type] += 1
        if ev.type == "response.output_text.delta":
            if first is None:
                first = time.perf_counter() - t.t0  # thời điểm token đầu tiên tới
            deltas.append(ev.delta)
        elif ev.type == "response.completed":
            final = ev.response
for name, n in events.items():
    print(f"{n:>3} x {name}")
print("first 6 deltas:", deltas[:6])
print("joined == final.output_text:", "".join(deltas) == final.output_text)
print(f"usage: input {final.usage.input_tokens}, output {final.usage.output_tokens} tokens")
print(f"time to first token: {first:.1f} s, total: {t.seconds:.1f} s, decode: {t.seconds - first:.1f} s")
print("answer:", final.output_text.strip())
print("model:", final.model)
