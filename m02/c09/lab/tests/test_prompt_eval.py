from string import Template

import pytest

from common.llm import FakeLLM
from prompt_eval import (LABELS, PROMPT_V1, PROMPT_V2, REPLY_PROMPT, check_reply, classify, compare,
                         evaluate, gate, parse_label, render)

ROWS = [
    {"message_id": "M1", "text": "hang vo roi, cho toi xin lai tien", "intent": "refund"},
    {"message_id": "M2", "text": "Đơn AT046354 khoảng mấy phút nữa thì tới", "intent": "eta"},
    {"message_id": "M3", "text": "giao hang cham nhu rua", "intent": "complaint_late"},
    {"message_id": "M4", "text": "shop co ban thung carton khong", "intent": "other"},
]


def test_render_fills_labels():
    text = render(PROMPT_V1)
    assert "$" not in text and all(label in text for label in LABELS)


def test_render_missing_variable_raises():
    with pytest.raises(KeyError):
        render(REPLY_PROMPT)


def test_render_keeps_json_braces():
    tpl = Template('Reply as JSON like {"label": "refund"}. Labels: $labels')
    assert '{"label": "refund"}' in render(tpl)


@pytest.mark.parametrize("text,label", [
    ("refund", "refund"), ("Refund.", "refund"), ("The label is eta.", "eta"),
    ("refund or complaint_late", "invalid"), ("", "invalid"), ("late", "invalid"),
])
def test_parse_label(text, label):
    assert parse_label(text) == label


def test_classify_sends_system_and_wrapped_message():
    fake = FakeLLM(["eta"])
    assert classify(fake, "INSTRUCTIONS", "mai may gio giao") == "eta"
    sent = fake.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[0]["content"] == "INSTRUCTIONS"
    assert sent[1]["content"] == "<message>mai may gio giao</message>"


def test_evaluate_counts():
    fake = FakeLLM(["refund", "The label is eta.", "eta", "no idea"])
    report = evaluate(fake, "X", ROWS)
    assert report["accuracy"] == 0.5
    assert report["invalid"] == 1
    assert report["per_intent"]["complaint_late"] == (0, 1)
    assert report["per_intent"]["eta"] == (1, 1)
    assert report["predictions"] == {"M1": "refund", "M2": "eta", "M3": "eta", "M4": "invalid"}


def test_compare_fixed_and_broken():
    a = {"M1": "refund", "M2": "order_status", "M3": "complaint_late", "M4": "other"}
    b = {"M1": "complaint_late", "M2": "eta", "M3": "complaint_late", "M4": "other"}
    assert compare(ROWS, a, b) == {"fixed": ["M2"], "broken": ["M1"]}


def test_check_reply_good():
    reply = "Mình xin lỗi vì đơn AT000021 bị trễ. Đội Giao Hàng Anh Tí sẽ kiểm tra và phản hồi bạn trong 24 giờ."
    assert check_reply(reply, "AT000021") == []


def test_check_reply_promise_and_missing_id():
    reply = "Xin lỗi bạn, mình sẽ hoàn tiền ngay cho bạn."
    assert check_reply(reply, "AT000021") == ["order_id", "no_promise"]


def test_check_reply_too_long():
    reply = "Xin lỗi. Đơn AT000021 trễ. Mình kiểm tra. Mình báo lại. Cảm ơn bạn."
    assert check_reply(reply, "AT000021") == ["short"]


def test_gate():
    old = {"accuracy": 0.70, "invalid": 1}
    assert gate(old, {"accuracy": 0.74, "invalid": 0}, {"fixed": ["M2", "M5"], "broken": ["M1"]}, max_broken=1) == (True, [])
    ok, reasons = gate(old, {"accuracy": 0.68, "invalid": 3}, {"fixed": [], "broken": ["M1", "M4"]}, max_broken=1)
    assert not ok and len(reasons) == 3


def test_prompt_v2_defines_every_label():
    v1, v2 = render(PROMPT_V1), render(PROMPT_V2)
    assert len(v2) > len(v1)
    assert all(v2.count(label) >= 2 for label in LABELS)
    assert "<message>" in v2
