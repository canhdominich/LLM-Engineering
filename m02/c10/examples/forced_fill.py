import os

from openai import OpenAI

from _delivery import DeliveryRequest

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=600)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
MESSAGES = [
    "Cảm ơn shop, hôm qua giao nhanh lắm",
    "Đơn AT036959 đang ở đâu vậy?",
]
for text in MESSAGES:
    resp = client.responses.parse(model=MODEL, instructions="Extract the delivery request from the customer message.",
                                  input=text, text_format=DeliveryRequest, temperature=0, max_output_tokens=300,
                                  reasoning={"effort": "none"})
    print(repr(text))
    print("   ", resp.output_parsed)
print("model:", MODEL)
