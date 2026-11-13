import numpy as np
import pytest

from common.llm import FakeLLM
from model_choice import call_cost_usd, evaluate, llm_classifier, percentile, pick, quality, summarize

PRICE = {"input": 0.10, "cached_input": 0.01, "output": 0.50}


def test_percentile_matches_numpy_and_keeps_input():
    xs = [5.0, 1.0, 4.0, 2.0, 3.0]
    assert percentile(xs, 50) == pytest.approx(3.0)
    assert percentile(xs, 95) == pytest.approx(float(np.percentile(xs, 95)))
    assert xs == [5.0, 1.0, 4.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_quality_counts_invalid_as_wrong_and_averages_all_labels():
    truth = ["refund", "refund", "eta", "other"]
    pred = ["refund", "invalid", "eta", "eta"]
    q = quality(truth, pred)
    assert q["accuracy"] == pytest.approx(0.5)
    assert q["invalid"] == 1
    # refund 2/3, eta 2/3, other 0, ba nhãn không xuất hiện cũng tính 0
    assert q["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 6)


def test_cached_tokens_use_cached_rate():
    assert call_cost_usd(1000, 0, 10, PRICE) == pytest.approx((1000 * 0.10 + 10 * 0.50) / 1e6)
    assert call_cost_usd(1000, 900, 10, PRICE) == pytest.approx((100 * 0.10 + 900 * 0.01 + 10 * 0.50) / 1e6)
    with pytest.raises(ValueError):
        call_cost_usd(100, 200, 1, PRICE)


def test_summarize_api_price_and_self_host():
    recs = [{"intent": "eta", "pred": "eta", "seconds": s, "in_tok": 300, "cached_tok": 200, "out_tok": 3}
            for s in (1.0, 2.0, 3.0, 4.0)]
    api = summarize("api", recs, price=PRICE)
    assert api["usd_per_1000"] == pytest.approx(1000 * call_cost_usd(300, 200, 3, PRICE))
    assert api["p50"] == pytest.approx(2.5) and api["p95"] == pytest.approx(3.85)
    local = summarize("local", recs, usd_per_hour=3.6)
    assert local["usd_per_1000"] == pytest.approx(1000 * 2.5 / 3600 * 3.6)
    assert local["accuracy"] == 1.0


def test_pick_cheapest_that_passes_both_thresholds():
    rows = [
        {"name": "big", "macro_f1": 0.90, "p95": 2.0, "usd_per_1000": 5.0},
        {"name": "small", "macro_f1": 0.80, "p95": 3.0, "usd_per_1000": 0.5},
        {"name": "tiny", "macro_f1": 0.60, "p95": 0.1, "usd_per_1000": 0.01},
        {"name": "slow", "macro_f1": 0.85, "p95": 9.0, "usd_per_1000": 0.2},
    ]
    assert pick(rows, min_f1=0.75, max_p95=5.0) == "small"
    assert pick(rows, min_f1=0.88, max_p95=5.0) == "big"
    assert pick(rows, min_f1=0.95, max_p95=5.0) is None


def test_pick_breaks_cost_tie_by_p95():
    rows = [{"name": "a", "macro_f1": 0.8, "p95": 4.0, "usd_per_1000": 1.0},
            {"name": "b", "macro_f1": 0.8, "p95": 2.0, "usd_per_1000": 1.0}]
    assert pick(rows, 0.75, 5.0) == "b"


def test_llm_classifier_with_fake_llm():
    llm = FakeLLM(["refund", "Label: eta.", "not sure, maybe refund or eta"])
    classify = llm_classifier(llm, instructions="Classify.")
    rows = [{"message_id": f"M{i}", "text": t, "intent": y}
            for i, (t, y) in enumerate([("tra lai tien", "refund"), ("khi nao toi", "eta"), ("hmm", "other")])]
    recs = evaluate(classify, rows)
    assert [r["pred"] for r in recs] == ["refund", "eta", "invalid"]
    assert all(r["seconds"] >= 0 and r["in_tok"] > 0 for r in recs)
    assert llm.calls[0]["messages"][0] == {"role": "system", "content": "Classify."}
    assert llm.calls[0]["messages"][1]["content"] == "<message>tra lai tien</message>"
    assert summarize("fake", recs, usd_per_hour=0.0)["accuracy"] == pytest.approx(2 / 3)
