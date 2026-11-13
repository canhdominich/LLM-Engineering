import time

import openai

from _setup import INSTRUCTIONS, MODEL, client

QUESTION = "Liệt kê chi tiết mọi bước xử lý một đơn giao trễ, mỗi bước một đoạn dài."
for retries in (0, 2):
    llm = client(timeout=2.0, max_retries=retries)  # 2 giây cho mỗi lần thử
    t0 = time.perf_counter()
    try:
        llm.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=QUESTION,
                             max_output_tokens=400, reasoning={"effort": "none"})
        print(f"max_retries={retries}: answered")
    except openai.APITimeoutError as e:
        print(f"max_retries={retries}: {type(e).__name__} after {time.perf_counter() - t0:.1f} s")
