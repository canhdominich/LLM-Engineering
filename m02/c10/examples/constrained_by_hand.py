import math
import os

import httpx

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
DISTRICTS = ["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]
TARGETS = [d + '"' for d in DISTRICTS]


def top_tokens(prompt):
    # một bước decode: xin đúng 1 token và 20 ứng viên xác suất cao nhất
    body = {"model": MODEL, "prompt": prompt, "raw": True, "stream": False, "logprobs": True, "top_logprobs": 20,
            "options": {"num_predict": 1, "temperature": 0}}
    r = httpx.post(URL, json=body, timeout=600).json()
    return [(c["token"], c["logprob"]) for c in r["logprobs"][0]["top_logprobs"]]


def allowed(text):
    # text = phần đã sinh + token ứng viên; hợp lệ nếu còn là tiền tố của một giá trị enum
    if '"' in text:
        return text[:text.index('"') + 1] in TARGETS
    return any(x.startswith(text) for x in TARGETS)


def decode(prompt, constrained, max_steps=8):
    done, trace = "", []
    for _ in range(max_steps):
        left = [x for x in TARGETS if x.startswith(done)]
        if constrained and len(left) == 1:
            trace.append(f"jump {left[0][len(done):]!r}")  # chỉ còn một đường: điền luôn, không gọi model
            done = left[0]
            break
        cands = top_tokens(prompt + done)
        if constrained:
            cands = [(t, lp) for t, lp in cands if allowed(done + t)]
        if not cands:
            trace.append("no allowed token in top 20")
            break
        token, lp = cands[0]
        trace.append(f"{token!r} p={math.exp(lp):.2f}")
        done += token
        if '"' in token:
            break
    return done.split('"')[0], trace


CASES = [("Đầu Phố nhận 4 gói từ Trà Sữa Cô Na ở Bãi Biển nha, thu 2000 nghìn", "pickup_district"),
         ("giao 9 thung tu cho moi qua ben song trong 1 tieng, khong thu tien", "dropoff_district")]
for text, field in CASES:
    prompt = f'Customer message: {text}\nJSON: {{"{field}": "'
    for constrained in (False, True):
        value, trace = decode(prompt, constrained)
        label = "masked" if constrained else "free  "
        print(f"{field} {label}: {value!r:20} valid={value in DISTRICTS}")
        print("    steps:", ", ".join(trace))
print("model:", MODEL)
