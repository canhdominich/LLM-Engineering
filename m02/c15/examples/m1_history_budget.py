import _project  # noqa: F401
from common.llm import FakeLLM
from conftest import AN, triage
from transformers import AutoTokenizer

from anhti_chat.chat import ChatService
from anhti_chat.config import load_settings
from anhti_chat.evals import load_conversations
from anhti_chat.history import build_context, estimate_tokens
from anhti_chat.llm import FakeClient
from anhti_chat.store import MemoryStore
from anhti_chat.tools import OrderBook, SnapshotEta, ToolRunner

case = next(c for c in load_conversations() if c["id"] == "E40")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
texts = [h[1] for h in case["history"]] + [case["message"]]
real = sum(len(tok(t)["input_ids"]) for t in texts)
est = sum(estimate_tokens(t) for t in texts)
print(f"E40: {len(texts)} messages, {sum(map(len, texts))} chars, Qwen3 tokens {real}, estimate {est} ({est / real:.2f}x)")

msgs = [{"seq": i + 1, "role": r, "content": c, "status": "complete"} for i, (r, c) in enumerate(case["history"])]
msgs.append({"seq": len(msgs) + 1, "role": "user", "content": case["message"], "status": "complete"})
for budget in (1000, 300, 60):
    items, older = build_context(msgs, 0, budget)
    print(f"budget {budget:>4}: send {len(items)} messages, fold {len(older)} older into the summary")

settings = load_settings(provider="fake")
store = MemoryStore()
llm = FakeClient(FakeLLM(["An có hai đơn: AT047838 tới Cho Moi (mẹ đã thấy shipper) và AT051122 tới Doi Thong.",
                          triage("eta", "AT051122"), "Bạn chờ mình tra nhé."]))
service = ChatService(llm, store, ToolRunner(OrderBook(settings.orders_path), SnapshotEta(settings.eta_snapshot), settings.clock), settings)
cid = store.create_conversation(AN)
for m in msgs[:-1]:
    store.add_message(cid, m["role"], m["content"])
list(service.handle(AN, cid, case["message"]))
conv = store.get_conversation(cid, AN)
print("summary_upto:", conv["summary_upto"], "| summary:", conv["summary"])
sent = llm.llm.calls[1]["messages"]
print("triage call got", len(sent) - 1, "messages; instructions end with:", sent[0]["content"].splitlines()[-1][:60] + "...")
print("usage kinds:", [u["kind"] for u in store.usage(cid)])
