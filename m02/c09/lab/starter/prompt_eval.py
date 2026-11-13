import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from string import Template

LABELS = ["order_status", "eta", "complaint_late", "refund", "change_address", "other"]
LABEL_RE = re.compile(r"\b(" + "|".join(LABELS) + r")\b")
FORBIDDEN = ["hoàn tiền", "hoan tien", "bồi thường", "boi thuong", "đền bù", "den bu", "phạt", "sa thải"]

PROMPT_V1 = Template("""Classify this customer message of a delivery company.
Reply with exactly one label from this list and nothing else: $labels.""")

# TODO 1: viết PROMPT_V2: giữ danh sách nhãn của v1, thêm mô tả dữ liệu, định nghĩa từng nhãn,
# quy tắc "trễ mà đòi tiền là refund" và câu nói tin nhắn nằm giữa thẻ <message>.
PROMPT_V2 = PROMPT_V1

REPLY_PROMPT = Template("""You are the support assistant of Giao Hang Anh Ti.
The customer complains that order $order_id is late.
Write the reply in Vietnamese, at most three sentences: apologise, repeat the order id,
say the team will check and answer within 24 hours.
Never promise money back, compensation or a penalty for the shipper.""")


def render(template, **values):
    # TODO 2: điền $labels bằng LABELS nối bởi ", " và các biến trong values; thiếu biến phải báo KeyError.
    raise NotImplementedError


def parse_label(text):
    # TODO 3: trả về nhãn khi văn bản chứa đúng một nhãn khác nhau (không phân biệt hoa thường), ngược lại "invalid".
    raise NotImplementedError


def classify(llm, instructions, message):
    # TODO 4: gửi [system: instructions, user: <message>...</message>] với max_tokens=60, rồi parse_label câu trả lời.
    raise NotImplementedError


def evaluate(llm, instructions, rows):
    # TODO 5: trả về dict accuracy, invalid, per_intent {intent: (đúng, tổng)}, predictions {message_id: nhãn}.
    raise NotImplementedError


def compare(rows, pred_a, pred_b):
    # TODO 6: fixed là các message_id sai ở a và đúng ở b, broken là đúng ở a và sai ở b; cả hai đã sắp xếp.
    raise NotImplementedError


def check_reply(reply, order_id):
    # TODO 7: trả về danh sách luật bị vi phạm theo thứ tự order_id, apology, no_promise, short.
    raise NotImplementedError


def gate(old, new, diff, max_broken=0):
    # TODO 8: trả về (True, []) khi accuracy không giảm, số broken không quá max_broken và invalid không tăng;
    # ngược lại (False, danh sách lý do), mỗi điều kiện vi phạm là một lý do.
    raise NotImplementedError


class ResponsesLLM:
    def __init__(self, model=None):
        from openai import OpenAI
        self.model = model or os.environ.get("LLM_MODEL", "qwen3:1.7b")
        self.client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                             api_key=os.environ.get("OPENAI_API_KEY", "ollama"), timeout=900)
        self.cache_dir = os.environ.get("LLM_CACHE_DIR")

    def chat(self, messages, max_tokens=60):
        key = hashlib.sha256(json.dumps([self.model, messages, max_tokens]).encode()).hexdigest()[:24]
        path = Path(self.cache_dir) / f"{key}.json" if self.cache_dir else None
        if path and path.exists():  # cùng prompt, cùng model: dùng lại câu trả lời đã trả tiền
            text = json.loads(path.read_text(encoding="utf-8"))["text"]
        else:
            r = self.client.responses.create(model=self.model, instructions=messages[0]["content"],
                                             input=messages[1]["content"], temperature=0,
                                             max_output_tokens=max_tokens, reasoning={"effort": "none"})
            text = r.output_text.strip()
            if path:
                path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
        return type("Reply", (), {"text": text})


def load_rows(n=50):
    import pandas as pd
    path = Path(os.environ.get("ANHTI_M02_DATA", Path(__file__).resolve().parents[3] / "data")) / "messages.csv"
    df = pd.read_csv(path)
    test = df[df["split"] == "test"].sort_values("message_id").head(n)
    return test[["message_id", "text", "intent"]].to_dict("records")


if __name__ == "__main__":
    rows = load_rows()
    llm = ResponsesLLM()
    reports = {}
    for name, tpl in [("v1", PROMPT_V1), ("v2", PROMPT_V2)]:
        reports[name] = evaluate(llm, render(tpl), rows)
        print(f"{name}: accuracy {reports[name]['accuracy']:.3f}, invalid {reports[name]['invalid']}")
    print(f"{'intent':16}{'v1':>7}{'v2':>7}")
    for intent in LABELS:
        a, b = reports["v1"]["per_intent"].get(intent, (0, 0)), reports["v2"]["per_intent"].get(intent, (0, 0))
        print(f"{intent:16}{a[0]:>4}/{a[1]:<2}{b[0]:>4}/{b[1]:<2}")
    diff = compare(rows, reports["v1"]["predictions"], reports["v2"]["predictions"])
    print(f"fixed {len(diff['fixed'])}, broken {len(diff['broken'])}: {diff['broken']}")
    ok, reasons = gate(reports["v1"], reports["v2"], diff, max_broken=2)
    print("gate:", "ship v2" if ok else "block v2", reasons)
    complaints = [r for r in rows if r["intent"] == "complaint_late"]
    passed = 0
    for r in complaints:
        order_id = f"AT{int(r['message_id'][1:]):06d}"  # đơn gần nhất của khách, do hệ thống biết
        reply = llm.chat([{"role": "system", "content": render(REPLY_PROMPT, order_id=order_id)},
                          {"role": "user", "content": r["text"]}], max_tokens=200).text
        failed = check_reply(reply, order_id)
        passed += not failed
        print(f"  {r['message_id']} {failed or 'ok'}: {reply[:70]!r}")
    print(f"replies passing all checks: {passed}/{len(complaints)}")
    print(f"model {llm.model}, {len(rows) * 2 + len(complaints)} requests")
