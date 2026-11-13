import os

from openai import OpenAI
from transformers import AutoTokenizer

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"), api_key="ollama", timeout=120)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
SYSTEM = "Bạn là trợ lý của Giao Hàng Anh Tí. Trả lời ngắn gọn, lịch sự, bằng tiếng Việt."

cases = [
    ("vi, no system", None, "Đơn AT027075 của tôi đang ở đâu vậy?"),
    ("en, no system", None, "Where is my order AT027075 right now?"),
    ("vi, with system", SYSTEM, "Đơn AT027075 của tôi đang ở đâu vậy?"),
]
print("model:", MODEL)
for name, system, text in cases:
    r = client.responses.create(model=MODEL, instructions=system, input=text,
                                max_output_tokens=8, reasoning={"effort": "none"})
    local = len(qwen.encode(text)) + (len(qwen.encode(system)) if system else 0)
    billed = r.usage.input_tokens
    print(f"{name:16} text tokens={local:3}  billed input={billed:3}  overhead={billed - local:3}")
