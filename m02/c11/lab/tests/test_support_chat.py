import json

import pytest
from common.llm import FakeLLM

import support_chat as sc


@pytest.fixture(scope="module")
def orders():
    return sc.load_orders()


@pytest.fixture(scope="module")
def registry(orders):
    return sc.make_registry(orders, sc.train_eta_model())


def call(i, name, order_id):
    return {"id": f"call_{i}", "name": name, "arguments": {"order_id": order_id}}


def test_tools_schema_is_strict_and_described():
    names = [t["function"]["name"] for t in sc.TOOLS]
    assert names == ["get_order_status", "predict_eta"]
    for t in sc.TOOLS:
        f = t["function"]
        params = f["parameters"]
        assert t["type"] == "function" and f["strict"] is True
        assert params["additionalProperties"] is False and params["required"] == ["order_id"]
        assert params["properties"]["order_id"]["pattern"] == r"^AT\d{6}$"
        assert len(f["description"]) >= 60
    assert sc.TOOLS[0]["function"]["description"] != sc.TOOLS[1]["function"]["description"]


def test_get_order_status(orders):
    assert sc.get_order_status("AT047838", orders) == {"order_id": "AT047838", "status": "out_for_delivery", "district": "Cho Moi"}
    assert sc.get_order_status("AT999999", orders)["error"] == "order_not_found"


def test_predict_eta_on_the_way_and_delivered(orders, registry):
    out = registry["predict_eta"]("AT047838")
    assert out == {"order_id": "AT047838", "minutes_left": 11}
    assert isinstance(out["minutes_left"], int)
    assert registry["predict_eta"]("AT039405")["error"] == "not_on_the_way"
    assert registry["predict_eta"]("AT060017")["error"] == "not_on_the_way"


def test_execute_call_never_raises(registry):
    assert sc.execute_call("cancel_order", {"order_id": "AT047838"}, registry)["error"] == "unknown_tool"
    assert sc.execute_call("get_order_status", {"order_id": "47838"}, registry)["error"] == "invalid_arguments"
    assert sc.execute_call("get_order_status", {"order_id": "AT047838", "force": True}, registry)["error"] == "invalid_arguments"
    assert sc.execute_call("get_order_status", '{"order_id": "AT047838"}', registry)["status"] == "out_for_delivery"

    def boom(order_id):
        raise ConnectionError("eta service down")

    out = sc.execute_call("predict_eta", {"order_id": "AT047838"}, {"predict_eta": boom})
    assert out["error"] == "tool_failed" and out["retryable"] is True and "ConnectionError" in out["message"]


def test_run_chat_happy_path(registry):
    llm = FakeLLM([{"tool_calls": [call(1, "get_order_status", "AT047838")]}, {"tool_calls": [call(2, "predict_eta", "AT047838")]},
                   "Don dang duoc giao, khoang 11 phut nua."])
    result = sc.run_chat(llm, "Don AT047838 dang o dau?", registry)
    assert result.stopped == "answer" and result.steps == 3
    assert result.answer == "Don dang duoc giao, khoang 11 phut nua."
    assert [c["name"] for c in result.tool_calls] == ["get_order_status", "predict_eta"]
    assert len(llm.calls) == 3
    assert all(c["tools"] == sc.TOOLS for c in llm.calls)


def test_messages_follow_chat_completions_format(registry):
    llm = FakeLLM([{"tool_calls": [call(7, "get_order_status", "AT047838")]}, "ok"])
    sc.run_chat(llm, "Don AT047838 dang o dau?", registry)
    second = llm.calls[1]["messages"]
    assert second[0]["role"] == "system" and second[1] == {"role": "user", "content": "Don AT047838 dang o dau?"}
    assistant, tool = second[2], second[3]
    assert assistant["role"] == "assistant"
    fn = assistant["tool_calls"][0]
    assert fn["id"] == "call_7" and fn["type"] == "function" and fn["function"]["name"] == "get_order_status"
    assert json.loads(fn["function"]["arguments"]) == {"order_id": "AT047838"}
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_7"
    assert json.loads(tool["content"])["status"] == "out_for_delivery"


def test_parallel_calls_keep_order(registry):
    llm = FakeLLM([{"tool_calls": [call(1, "get_order_status", "AT047838"), call(2, "get_order_status", "AT051122")]}, "ok"])
    sc.run_chat(llm, "Hai don dang o dau?", registry)
    msgs = llm.calls[1]["messages"]
    tools = [m for m in msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["call_1", "call_2"]
    assert [json.loads(m["content"])["status"] for m in tools] == ["out_for_delivery", "picked_up"]
    assert sum(1 for m in msgs if m["role"] == "assistant") == 1


def test_tool_error_goes_back_to_model(registry):
    llm = FakeLLM([{"tool_calls": [call(1, "get_order_status", "AT999999")]}, "Khong tim thay don."])
    result = sc.run_chat(llm, "Don AT999999?", registry)
    assert result.stopped == "answer"
    tool = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool["content"])["error"] == "order_not_found"


def test_max_steps_stops_the_loop(registry):
    ids = ["AT047838", "AT051122", "AT052290", "AT039405", "AT060017", "AT047838"]
    llm = FakeLLM([{"tool_calls": [call(i, "get_order_status", oid)]} for i, oid in enumerate(ids)] * 2)
    result = sc.run_chat(llm, "?", registry, max_steps=3)
    assert result.stopped == "max_steps" and result.steps == 3
    assert len(llm.calls) == 3
    assert result.answer == sc.FALLBACK


def test_repeated_call_stops_early(registry):
    llm = FakeLLM([{"tool_calls": [call(1, "predict_eta", "AT039405")]}, {"tool_calls": [call(2, "predict_eta", "AT039405")]}, "never"])
    result = sc.run_chat(llm, "Don AT039405 bao gio toi?", registry, max_steps=4)
    assert result.stopped == "repeated_call" and result.steps == 2
    assert len(llm.calls) == 2
    assert result.answer == sc.FALLBACK


def test_direct_answer_without_tools(registry):
    llm = FakeLLM(["Chao ban, minh co the giup gi?"])
    result = sc.run_chat(llm, "Xin chao", registry)
    assert result.stopped == "answer" and result.steps == 1 and result.tool_calls == []
