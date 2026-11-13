import os
import time

from openai import OpenAI

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=120)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")

MESSAGE = "Đơn AT036959 trễ quá, hủy đi và trả lại tiền cho tôi"
TASKS = {  # cùng một model, mỗi tác vụ chỉ khác chỉ dẫn
    "intent": "Classify the message. Reply with one label: order_status, eta, complaint_late, refund, change_address, other.",
    "order_id": "Extract the order id (AT followed by 6 digits). Reply with the id only, or NONE.",
    "sentiment": "Is the customer's mood positive, neutral or negative? Reply with one word.",
    "reply": "You are a support agent of Giao Hang Anh Ti. Write one short polite reply in Vietnamese.",
}

for task, instructions in TASKS.items():
    t0 = time.perf_counter()
    resp = client.responses.create(model=MODEL, instructions=instructions, input=MESSAGE, temperature=0,
                                   max_output_tokens=700, reasoning={"effort": "none"})
    answer = " ".join(resp.output_text.split())
    print(f"[{task}] {time.perf_counter() - t0:.1f}s, {resp.usage.output_tokens} output tokens")
    print(f"  {answer[:150]}")
