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
    model_config = ConfigDict(extra="forbid")

    pickup_district: District = Field(description="District where the shipper picks up the goods")
    dropoff_district: District = Field(description="District where the goods are delivered")
    items: int = Field(ge=1, le=30, description="Number of packages")
    cod_vnd: int = Field(ge=0, description="Cash to collect from the receiver in VND, 0 if prepaid")
    service: Literal["express", "standard"] = Field(description="express only if the customer asks for urgent delivery")

    @field_validator("cod_vnd")
    @classmethod
    def cod_in_vnd(cls, v):
        if v % 1000:
            raise ValueError(f"must be in VND and a multiple of 1000, got {v}; 250k means 250000")
        if v > COD_LIMIT:
            raise ValueError(f"{v} is above the COD limit of {COD_LIMIT} VND")
        return v


def response_format():
    return {"type": "json_schema",
            "json_schema": {"name": "delivery_request", "schema": DeliveryRequest.model_json_schema(), "strict": True}}


@dataclass
class ExtractResult:
    request: Optional[DeliveryRequest]
    attempts: int
    errors: list = field(default_factory=list)


def error_text(err):
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'request'}: {e['msg']}" for e in err.errors())


def extract(text, llm, max_attempts=3):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    errors = []
    for attempt in range(1, max_attempts + 1):
        reply = llm.chat(messages, response_format=response_format(), max_tokens=300).text
        try:
            return ExtractResult(DeliveryRequest.model_validate_json(reply), attempt, errors)
        except ValidationError as e:
            errors.append(error_text(e))
            messages = messages + [{"role": "assistant", "content": reply},
                                   {"role": "user", "content": f"Your JSON is invalid: {errors[-1]}. Fix it and answer again."}]
    return ExtractResult(None, max_attempts, errors)


def field_accuracy(preds, golds):
    if len(preds) != len(golds):
        raise ValueError(f"{len(preds)} predictions for {len(golds)} gold rows")
    n = max(1, len(golds))
    out = {}
    for f in FIELDS:
        out[f] = sum(p is not None and getattr(p, f) == g[f] for p, g in zip(preds, golds)) / n
    out["all_fields"] = sum(p is not None and all(getattr(p, f) == g[f] for f in FIELDS) for p, g in zip(preds, golds)) / n
    return out


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
