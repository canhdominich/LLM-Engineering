import csv
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DATA = Path(os.environ.get("ANHTI_C10_DATA", Path(__file__).resolve().parents[2] / "data"))
DISTRICTS = ("Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep")
District = Literal["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]
FIELDS = ["pickup_district", "dropoff_district", "items", "cod_vnd", "service"]
COD_LIMIT = 5_000_000
SYSTEM = ("You extract delivery requests for Giao Hang Anh Ti in the city Pho Rom Ra. "
          "Districts use ASCII names without diacritics. cod_vnd is the cash the receiver pays in VND "
          "(250k means 250000, 1tr2 means 1200000), 0 if prepaid. service is express only for urgent requests.")


class DeliveryRequest(BaseModel):
    # TODO 1: cấm khóa lạ (extra="forbid"); hai quận dùng kiểu District; items từ 1 đến 30;
    # cod_vnd >= 0, là bội của 1000 và không quá COD_LIMIT (dùng field_validator);
    # service chỉ nhận "express" hoặc "standard". Thêm description cho từng trường.
    pickup_district: str
    dropoff_district: str
    items: int
    cod_vnd: int
    service: str


def response_format():
    # TODO 2: trả về dict response_format kiểu "json_schema" của Chat Completions:
    # name "delivery_request", schema lấy từ DeliveryRequest, strict True
    return {"type": "json_object"}


@dataclass
class ExtractResult:
    request: Optional[DeliveryRequest]
    attempts: int
    errors: list = field(default_factory=list)


def error_text(err):
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'request'}: {e['msg']}" for e in err.errors())


def extract(text, llm, max_attempts=3):
    # TODO 3: gửi [system SYSTEM, user text] kèm response_format=response_format(), max_tokens=300;
    # validate bằng DeliveryRequest.model_validate_json; nếu ValidationError thì ghi error_text(e) vào errors,
    # thêm câu trả lời sai (assistant) và lỗi (user) vào hội thoại rồi thử lại, tối đa max_attempts lần.
    reply = llm.chat([{"role": "user", "content": text}]).text
    try:
        return ExtractResult(DeliveryRequest.model_validate_json(reply), 1, [])
    except ValidationError as e:
        return ExtractResult(None, 1, [error_text(e)])


def field_accuracy(preds, golds):
    # TODO 4: báo ValueError khi hai danh sách khác độ dài; với mỗi trường trong FIELDS tính tỉ lệ dự đoán đúng,
    # thêm khóa "all_fields" (đúng cả 5 trường); dự đoán None tính là sai ở mọi trường.
    return {f: 0.0 for f in FIELDS}


def load_gold(path=None):
    rows = list(csv.DictReader(open(path or DATA / "delivery_requests.csv", encoding="utf-8")))
    return [{**r, "items": int(r["items"]), "cod_vnd": int(r["cod_vnd"])} for r in rows]


class ResponsesLLM:
    """Adapter: same chat() interface as FakeLLM, but calls the Responses API without reasoning."""

    def __init__(self, model=None):
        from openai import OpenAI
        self.model = model or os.environ.get("LLM_MODEL", "qwen3:1.7b")
        self.client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                             api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=600)

    def chat(self, messages, response_format=None, max_tokens=300):
        from common.llm import Reply, Usage
        system = [m["content"] for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        kw = {}
        if response_format:
            js = response_format["json_schema"]
            kw["text"] = {"format": {"type": "json_schema", "name": js["name"], "schema": js["schema"], "strict": js["strict"]}}
        r = self.client.responses.create(model=self.model, instructions="\n".join(system) or None, input=rest,
                                         temperature=0, max_output_tokens=max_tokens, reasoning={"effort": "none"}, **kw)
        return Reply(text=r.output_text, usage=Usage(r.usage.input_tokens, r.usage.output_tokens), model=self.model)


if __name__ == "__main__":
    gold = load_gold()
    llm = ResponsesLLM()
    t0 = time.perf_counter()
    results = [extract(g["text"], llm) for g in gold]
    preds = [r.request for r in results]
    acc = field_accuracy(preds, gold)
    print(f"{len(gold)} messages, model {llm.model}, {time.perf_counter() - t0:.0f} s")
    for k, v in acc.items():
        print(f"  {k:17} {v:.3f}")
    print("attempts:", dict(sorted(Counter(r.attempts for r in results).items())),
          "failed:", sum(r.request is None for r in results))
    shown = 0
    for g, r in zip(gold, results):
        wrong = [f for f in FIELDS if r.request is None or getattr(r.request, f) != g[f]]
        if (wrong or r.errors) and shown < 12:
            shown += 1
            got = {f: getattr(r.request, f) for f in wrong} if r.request else None
            print(f"  {g['request_id']} wrong={wrong} got={got} retries={len(r.errors)}")
