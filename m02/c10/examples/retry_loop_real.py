import os

from openai import OpenAI

from _rules import Extraction, extract_with_retry

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=600)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
FORMAT = {"type": "json_schema", "name": "extraction", "schema": Extraction.model_json_schema(), "strict": True}


def call(messages):
    resp = client.responses.create(model=MODEL, instructions="Extract the delivery request from the customer message.",
                                   input=messages, text={"format": FORMAT}, temperature=0,
                                   max_output_tokens=300, reasoning={"effort": "none"})
    return resp.output_text


MESSAGES = [
    "Chuyển hai túi từ Đồi Thông về Xóm Vui giúp tôi, tiền hàng 2,5 triệu người nhận trả, không cần nhanh",
    "Hỏa tốc: Tạp Hóa Cô Tấm (Bến Sông) gửi bốn hộp đến Bãi Biển, thu hộ 50k",
    "giao giúp tôi 2 thùng tới Chợ Mới, thu 300k",
    "Cảm ơn shop, hôm qua giao nhanh lắm",
]
for text in MESSAGES:
    result, attempts, errors = extract_with_retry(call, text)
    print(repr(text[:60]))
    print(f"    attempts={attempts} errors={[e[:70] for e in errors]}")
    if result:
        print("   ", result.model_dump(exclude={"missing"}), "missing:", result.missing)
print("model:", MODEL)
