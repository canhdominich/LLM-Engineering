import os

from openai import OpenAI
from pydantic import model_validator

from _rules import FIELDS, Extraction, extract_with_retry

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=600)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")


class ExtractionV2(Extraction):
    @model_validator(mode="after")
    def every_status_has_rules(self):
        filled = [f for f in FIELDS if getattr(self, f) is not None]
        if self.status == "not_request" and filled:
            raise ValueError(f"status is not_request but {filled} are filled; set them to null")
        if self.status == "need_info" and not self.missing:
            raise ValueError("status is need_info but missing is empty")
        return self


INSTRUCTIONS = """Extract a delivery request for Giao Hang Anh Ti.
- not_request: the customer is not asking to send goods (thanks, questions about an existing order). All fields null.
- need_info: asking to send goods but the pickup or dropoff district, items, or cod is not stated. Never guess them: put null and list the names in missing.
- ok: every field is stated. The district after "từ" or "ở" of the sender is pickup_district; the place goods go to is dropoff_district."""
FORMAT = {"type": "json_schema", "name": "extraction", "schema": ExtractionV2.model_json_schema(), "strict": True}


def call(messages):
    resp = client.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=messages, text={"format": FORMAT},
                                   temperature=0, max_output_tokens=300, reasoning={"effort": "none"})
    return resp.output_text


MESSAGES = [
    "Chuyển hai túi từ Đồi Thông về Xóm Vui giúp tôi, tiền hàng 2,5 triệu người nhận trả, không cần nhanh",
    "Hỏa tốc: Tạp Hóa Cô Tấm (Bến Sông) gửi bốn hộp đến Bãi Biển, thu hộ 50k",
    "giao giúp tôi 2 thùng tới Chợ Mới, thu 300k",
    "Cảm ơn shop, hôm qua giao nhanh lắm",
]
for text in MESSAGES:
    result, attempts, errors = extract_with_retry(call, text, schema=ExtractionV2)
    print(repr(text[:60]))
    print(f"    attempts={attempts} errors={[e[:70] for e in errors]}")
    if result:
        print("   ", result.model_dump(exclude={"missing"}), "missing:", result.missing)
print("model:", MODEL)
