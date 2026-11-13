import json

from common.llm import FakeLLM
from _rules import extract_with_retry

good = {"status": "ok", "pickup_district": "Ben Song", "dropoff_district": "Bai Bien", "items": 4,
        "cod_vnd": 50000, "service": "express", "missing": []}
# kịch bản: lần 1 model đọc "50k" thành 50, lần 2 sửa đúng
fake = FakeLLM([json.dumps({**good, "cod_vnd": 50}), json.dumps(good)])
text = "Hỏa tốc: Tạp Hóa Cô Tấm (Bến Sông) gửi bốn hộp đến Bãi Biển, thu hộ 50k"

result, attempts, errors = extract_with_retry(lambda m: fake.chat(m).text, text)
print("attempts:", attempts)
print("errors:", errors)
print("result:", result.cod_vnd, result.status)
print("messages sent on attempt 2:", len(fake.calls[1]["messages"]))
print("last message:", fake.calls[1]["messages"][-1]["content"][:100])

fake = FakeLLM([json.dumps({**good, "cod_vnd": 50})] * 3)  # model sai mãi
result, attempts, errors = extract_with_retry(lambda m: fake.chat(m).text, text)
print("always wrong:", result, attempts, len(errors), "errors")
