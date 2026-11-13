import pytest

import inference_bench as ib

STREAM = [(0.40, "meta"), (1.20, "text"), (1.30, "text"), (1.40, "text"), (1.50, "text")]
THINKING = [(0.75, "reasoning"), (0.90, "reasoning"), (1.05, "reasoning"), (2.00, "text"), (2.25, "text")]
OLLAMA = {"load_duration": 500_000_000, "prompt_eval_count": 558, "prompt_eval_cached_count": 0,
          "prompt_eval_duration": 27_900_000_000, "eval_count": 40, "eval_duration": 5_000_000_000}


def test_ttft_is_first_visible_text_not_first_event():
    s = ib.summarize_events(STREAM)
    assert s["first_event_s"] == pytest.approx(0.40)
    assert s["ttft_s"] == pytest.approx(1.20)
    assert s["total_s"] == pytest.approx(1.50)


def test_tpot_and_decode_rate():
    s = ib.summarize_events(STREAM)
    assert s["visible_tokens"] == 4
    assert s["tpot_ms"] == pytest.approx(100.0)
    assert s["decode_tps"] == pytest.approx(10.0)


def test_reasoning_tokens_counted_and_delay_ttft():
    s = ib.summarize_events(THINKING)
    assert s["reasoning_tokens"] == 3
    assert s["visible_tokens"] == 2
    assert s["ttft_s"] == pytest.approx(2.00)
    assert s["tpot_ms"] == pytest.approx(250.0)


def test_no_visible_text_gives_none():
    s = ib.summarize_events([(0.7, "reasoning"), (0.8, "reasoning")])
    assert s["ttft_s"] is None
    assert s["tpot_ms"] is None
    assert s["decode_tps"] is None


def test_single_text_token_has_no_tpot():
    s = ib.summarize_events([(0.5, "text")])
    assert s["ttft_s"] == pytest.approx(0.5)
    assert s["tpot_ms"] is None


def test_from_ollama_converts_nanoseconds():
    r = ib.from_ollama(OLLAMA)
    assert r["load_s"] == pytest.approx(0.5)
    assert r["prefill_s"] == pytest.approx(27.9)
    assert r["prefill_tps"] == pytest.approx(20.0)
    assert r["decode_tps"] == pytest.approx(8.0)
    assert r["cached_tokens"] == 0


def test_percentile_linear_interpolation():
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert ib.percentile(xs, 50) == pytest.approx(3.0)
    assert ib.percentile(xs, 95) == pytest.approx(4.8)
    assert ib.percentile([2.0], 95) == pytest.approx(2.0)
    assert xs == [5.0, 1.0, 3.0, 2.0, 4.0]


def test_percentile_rejects_empty_and_bad_p():
    with pytest.raises(ValueError):
        ib.percentile([], 50)
    with pytest.raises(ValueError):
        ib.percentile([1.0], 120)


def test_memory_formulas():
    assert ib.weights_gb(2.03e9, 16) == pytest.approx(4.06)
    assert ib.weights_gb(2.03e9, 4) == pytest.approx(1.015)
    assert ib.kv_cache_gib(28, 8, 128, 4096) == pytest.approx(0.4375)
    assert ib.kv_cache_gib(28, 8, 128, 4096, users=8) == pytest.approx(3.5)


def test_summarize_runs_uses_median_and_counts_missing_text():
    runs = [ib.summarize_events(STREAM), ib.summarize_events(THINKING),
            ib.summarize_events([(0.7, "reasoning")])]
    s = ib.summarize_runs(runs)
    assert s["runs"] == 3
    assert s["no_text"] == 1
    assert s["ttft_p50"] == pytest.approx(1.60)
    assert s["decode_tps_p50"] == pytest.approx(7.0)
