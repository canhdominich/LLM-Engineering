import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=600)
SYSTEM = "Bạn là trợ lý của Giao Hàng Anh Tí. Trả lời tối đa hai câu, tiếng Việt có dấu."
QUESTION = "Đơn hỏa tốc đặt lúc 16 giờ ở quận Xóm Vui thì khi nào tới?"


def measure(effort, max_tokens):
    t0 = time.perf_counter()
    first_event = first_text = last_text = None
    n_reason = n_text = 0
    stream = client.responses.create(model="qwen3:1.7b", instructions=SYSTEM, input=QUESTION, stream=True,
                                     reasoning={"effort": effort}, max_output_tokens=max_tokens, temperature=0)
    for ev in stream:
        now = time.perf_counter() - t0
        first_event = first_event or now
        if ev.type == "response.reasoning_summary_text.delta":
            n_reason += 1  # token suy nghĩ: người dùng không thấy
        elif ev.type == "response.output_text.delta":
            first_text = first_text or now  # người dùng thấy chữ đầu tiên ở đây
            last_text, n_text = now, n_text + 1
        elif ev.type in ("response.completed", "response.incomplete"):
            status, text = ev.response.status, ev.response.output_text
    return first_event, first_text, last_text, n_reason, n_text, status, text


for effort, max_tokens in [("none", 80), ("medium", 400)]:
    first_event, first_text, last_text, n_reason, n_text, status, text = measure(effort, max_tokens)
    print(f"reasoning={effort}, max_output_tokens={max_tokens}, status={status}")
    sec = lambda x: "never" if x is None else f"{x:.2f}s"
    print(f"  first event {sec(first_event)}, first visible text {sec(first_text)}, last text {sec(last_text)}")
    print(f"  reasoning deltas {n_reason}, visible deltas {n_text}")
    print(f"  answer: {text.strip()[:150]}")
