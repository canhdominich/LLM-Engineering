import json
import re
import tomllib
from pathlib import Path

from .chat import ChatService
from .config import DEMO_USERS, EVALS_DIR, load_settings
from .guard import MINUTES_RE, ORDER_RE, PROMISES, fold, has
from .llm import Completed, ReplayClient, TextDelta, make_client
from .store import MemoryStore
from .tools import OrderBook, SnapshotEta, ToolRunner

USERS = {name.lower(): cust for cust, name in DEMO_USERS.values()}


def load_conversations(path=EVALS_DIR / "conversations.jsonl") -> list[dict]:
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def load_thresholds(path=EVALS_DIR / "thresholds.toml") -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


class Meter:
    """Wraps a client and keeps every Completed event, so latency and tokens come from the recording."""

    def __init__(self, inner):
        self.inner, self.name, self.model = inner, inner.name, inner.model
        self.calls = []

    def stream(self, instructions, items, tools=None, schema=None, max_output_tokens=300):
        first_text = False
        for ev in self.inner.stream(instructions, items, tools, schema, max_output_tokens):
            if isinstance(ev, TextDelta):
                first_text = True
            if isinstance(ev, Completed):
                self.calls.append((ev, first_text and schema is None))
            yield ev


def user_ttft(calls) -> float:
    total = 0.0
    for done, text in calls:
        if text:
            return round(total + (done.ttft or done.seconds), 2)
        total += done.seconds
    return round(total, 2)


def score(case: dict, events: list, owner_of) -> dict:
    exp, user = case["expected"], USERS[case["user"]]
    answer = "".join(e.data["text"] for e in events if e.name == "delta")
    for e in events:
        if e.name == "replace":
            answer = e.data["text"]
    folded = fold(answer)
    triage = next((e.data for e in events if e.name == "triage"), {})
    tools = [e.data for e in events if e.name == "tool"]
    handoff = any(e.name == "handoff" for e in events)
    r = {"id": case["id"], "answer": answer, "intent": triage.get("intent"), "handoff": handoff,
         "tools": [(t["name"], t["arguments"].get("order_id")) for t in tools],
         "replaced": next((e.data["reason"] for e in events if e.name == "replace"), None)}
    # TODO 20: điền intent_ok, handoff_ok, tools_ok (khi có expected tools), facts_ok (must và con số phút của predict_eta),
    # ungrounded (các số phút không có nguồn) và violations (forbid, promise, tool_leak, foreign_id). Xem bài Mốc 4.
    raise NotImplementedError("TODO 20")


def aggregate(rows: list[dict]) -> dict:
    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    def pct(vals, q):
        s = sorted(vals)
        return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else None

    ttfts = [r["ttft"] for r in rows]
    return {"n": len(rows), "intent_accuracy": rate("intent_ok"), "tool_accuracy": rate("tools_ok"),
            "handoff_accuracy": rate("handoff_ok"), "fact_accuracy": rate("facts_ok"),
            "safety_violations": sum(len(r["violations"]) for r in rows),
            "ungrounded_minutes": sum(len(r["ungrounded"]) for r in rows),
            "guard_replacements": sum(1 for r in rows if r["replaced"]),
            "ttft_p50_s": pct(ttfts, 0.5), "ttft_p95_s": pct(ttfts, 0.95),
            "llm_calls_per_conv": round(sum(r["calls"] for r in rows) / len(rows), 2),
            "tokens_per_conv": round(sum(r["tokens"] for r in rows) / len(rows), 1)}


def check_gate(metrics: dict, thresholds: dict) -> list[str]:
    # TODO 21: với mỗi khóa trong thresholds["min"] và thresholds["max"], ghi một dòng "key value < low" hoặc "key value > high".
    return []


def run_eval(record=False, only=None, client=None, cases=None, cache_dir=EVALS_DIR / "cache", settings=None, guard=True) -> dict:
    settings = settings or load_settings()
    if client is None:
        client = ReplayClient(make_client(settings) if record else None, cache_dir, record=record)
    orders = OrderBook(settings.orders_path)
    tools = ToolRunner(orders, SnapshotEta(settings.eta_snapshot), settings.clock)
    rows = []
    for case in cases or load_conversations():
        if only and case["id"] not in only:
            continue
        meter, store = Meter(client), MemoryStore()
        service = ChatService(meter, store, tools, settings, guard=guard)
        user = USERS[case["user"]]
        cid = store.create_conversation(user)
        for role, content in case["history"]:
            store.add_message(cid, role, content)
        events = list(service.handle(user, cid, case["message"]))
        row = score(case, events, orders.owner)
        row["calls"] = len(meter.calls)
        row["tokens"] = sum(d.input_tokens + d.output_tokens for d, _ in meter.calls)
        row["ttft"] = user_ttft(meter.calls)
        row["error"] = next((e.data["code"] for e in events if e.name == "error"), None)
        rows.append(row)
    return {"model": client.model, "metrics": aggregate(rows), "rows": rows}
