import _project  # noqa: F401
from common.llm import FakeLLM
from conftest import AN, call, triage

from anhti_chat.chat import ChatService
from anhti_chat.config import load_settings
from anhti_chat.llm import FakeClient
from anhti_chat.store import MemoryStore
from anhti_chat.tools import OrderBook, SnapshotEta, ToolRunner

settings = load_settings(provider="fake")
runner = ToolRunner(OrderBook(settings.orders_path), SnapshotEta(settings.eta_snapshot), settings.clock)
cases = {
    "loop": [triage("order_status")] + [{"tool_calls": [call("get_order_status", f"AT00000{i}", i)]} for i in range(1, 9)],
    "promise": [triage("complaint_late", "AT047838"), "Rất xin lỗi An, công ty sẽ bồi thường cho bạn một mã giảm giá 50%."],
    "guess": [triage("order_status", "AT047838"), {"tool_calls": [call("get_order_status", "AT047838")]},
              "Đơn AT047838 đang được giao ở Cho Moi và sẽ đến trong khoảng 15 phút nữa."],
    "grounded": [triage("eta", "AT047838"), {"tool_calls": [call("predict_eta", "AT047838")]}, "Đơn AT047838 còn khoảng 11 phút nữa tới."],
    "refund": [triage("refund", "AT047838")],
}
for name, script in cases.items():
    store, llm = MemoryStore(), FakeClient(FakeLLM(script))
    events = list(ChatService(llm, store, runner, settings).handle(AN, None, "..."))
    done = events[-1].data
    replaced = next((e.data["reason"] for e in events if e.name == "replace"), "-")
    cid = events[0].data["conversation_id"]
    print(f"{name:<9} stopped={done['stopped']:<6} steps={done['steps']} llm_calls={done['calls']} tools={sum(e.name == 'tool' for e in events)} "
          f"replaced={replaced} handoffs={len(store.handoffs(cid))} saved={store.messages(cid)[-1]['status']}")
