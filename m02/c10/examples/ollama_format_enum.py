import json
import math
import os

import httpx

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
DISTRICTS = ["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]
TEXT = "giao 2 hộp từ Chợ Mới ra sân bay, thu 100k"  # sân bay không thuộc 8 quận


def ask(enum, system):
    schema = {"type": "object", "properties": {"pickup_district": {"type": "string", "enum": DISTRICTS},
                                               "dropoff_district": {"type": "string", "enum": enum}},
              "required": ["pickup_district", "dropoff_district"], "additionalProperties": False}
    body = {"model": MODEL, "stream": False, "think": False, "format": schema, "logprobs": True, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": TEXT}]}
    r = httpx.post(URL, json=body, timeout=600).json()
    text, pos, probs = r["message"]["content"], 0, []
    start = text.index('"', text.index(":", text.index('"dropoff_district"'))) + 1
    end = text.index('"', start)
    for t in r["logprobs"]:  # xác suất model gán cho từng token của giá trị dropoff
        if start <= pos < end or (pos < start < pos + len(t["token"])):
            probs.append(f"{t['token']!r} {math.exp(t['logprob']):.2f}")
        pos += len(t["token"])
    return json.loads(text), probs

BASE = "Extract the pickup and dropoff districts."
HINT = BASE + " Use outside_city when a place is not one of the 8 districts of Pho Rom Ra."
CASES = [("enum 8", DISTRICTS, BASE), ("enum 8 + outside_city", DISTRICTS + ["outside_city"], BASE),
         ("enum 9 + told when to use", DISTRICTS + ["outside_city"], HINT)]
for name, enum, system in CASES:
    obj, probs = ask(enum, system)
    print(f"{name:26} {obj['pickup_district']} > {obj['dropoff_district']}")
    print("    dropoff tokens:", ", ".join(probs))
print("model:", MODEL)
