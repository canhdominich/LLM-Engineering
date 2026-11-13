import json

import pytest
from pydantic import ValidationError

from common.llm import FakeLLM
from delivery_extract import (DISTRICTS, FIELDS, DeliveryRequest, extract, field_accuracy, load_gold,
                              response_format)

GOOD = {"pickup_district": "Ben Song", "dropoff_district": "Bai Bien", "items": 4, "cod_vnd": 50000, "service": "express"}
TEXT = "Hỏa tốc: Tạp Hóa Cô Tấm (Bến Sông) gửi bốn hộp đến Bãi Biển, thu hộ 50k"


def test_schema_has_enum_required_and_no_extra():
    s = DeliveryRequest.model_json_schema()
    assert s["properties"]["pickup_district"]["enum"] == list(DISTRICTS)
    assert s["properties"]["dropoff_district"]["enum"] == list(DISTRICTS)
    assert s["properties"]["service"]["enum"] == ["express", "standard"]
    assert sorted(s["required"]) == sorted(FIELDS)
    assert s["additionalProperties"] is False


@pytest.mark.parametrize("change", [
    {"cod_vnd": 50}, {"cod_vnd": 6_000_000}, {"cod_vnd": -1000}, {"items": 0}, {"items": 31},
    {"pickup_district": "Quận Bến Sông"}, {"service": "fast"}, {"note": "nha"}, {"items": "bốn"},
])
def test_rejects_bad_values(change):
    with pytest.raises(ValidationError):
        DeliveryRequest.model_validate({**GOOD, **change})


def test_accepts_good_values():
    req = DeliveryRequest.model_validate({**GOOD, "cod_vnd": 0})
    assert req.cod_vnd == 0 and req.items == 4


def test_response_format_is_strict_json_schema():
    rf = response_format()
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"]
    assert rf["json_schema"]["schema"] == DeliveryRequest.model_json_schema()


def test_extract_first_try_sends_system_and_format():
    fake = FakeLLM([json.dumps(GOOD)])
    r = extract(TEXT, fake)
    assert r.request == DeliveryRequest(**GOOD)
    assert r.attempts == 1 and r.errors == []
    call = fake.calls[0]
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][-1] == {"role": "user", "content": TEXT}
    assert call["response_format"] == response_format()


def test_extract_retry_feeds_back_reply_and_error():
    bad = json.dumps({**GOOD, "cod_vnd": 50})
    fake = FakeLLM([bad, json.dumps(GOOD)])
    r = extract(TEXT, fake)
    assert r.attempts == 2 and r.request.cod_vnd == 50000
    assert len(r.errors) == 1 and "cod_vnd" in r.errors[0]
    second = fake.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": bad}
    assert second[-1]["role"] == "user" and "cod_vnd" in second[-1]["content"]
    assert len(fake.calls[0]["messages"]) == 2


def test_extract_retries_on_broken_json():
    fake = FakeLLM(["```json\n" + json.dumps(GOOD) + "\n```", json.dumps(GOOD)])
    r = extract(TEXT, fake)
    assert r.attempts == 2 and r.request is not None


def test_extract_stops_after_budget():
    fake = FakeLLM([json.dumps({**GOOD, "cod_vnd": 50})] * 5)
    r = extract(TEXT, fake, max_attempts=3)
    assert r.request is None and r.attempts == 3
    assert len(r.errors) == 3 and len(fake.calls) == 3


def test_field_accuracy_counts_none_as_wrong():
    golds = [dict(GOOD), dict(GOOD, cod_vnd=0), dict(GOOD)]
    preds = [DeliveryRequest(**GOOD), DeliveryRequest(**GOOD), None]
    acc = field_accuracy(preds, golds)
    assert acc["pickup_district"] == pytest.approx(2 / 3)
    assert acc["cod_vnd"] == pytest.approx(1 / 3)
    assert acc["all_fields"] == pytest.approx(1 / 3)
    assert set(acc) == set(FIELDS) | {"all_fields"}


def test_field_accuracy_length_mismatch():
    with pytest.raises(ValueError):
        field_accuracy([None], [])


def test_load_gold_types():
    gold = load_gold()
    assert len(gold) == 40
    assert all(isinstance(g["items"], int) and isinstance(g["cod_vnd"], int) for g in gold)
    assert {g["pickup_district"] for g in gold} <= set(DISTRICTS)
