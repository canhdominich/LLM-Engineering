import math
import os
import statistics
import time

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
MODELS = os.environ.get("BENCH_MODELS", "smollm2:135m-instruct-q8_0,smollm2:135m-instruct-q4_K_M").split(",")
PROMPT = "A customer asks: my parcel AT027075 is late, what should I do? Answer in two short sentences."


def summarize_events(events):
    first_event = events[0][0] if events else None
    text_times = [t for t, kind in events if kind == "text"]
    n_reason = sum(1 for _, kind in events if kind == "reasoning")
    ttft = text_times[0] if text_times else None
    tpot = (text_times[-1] - text_times[0]) / (len(text_times) - 1) if len(text_times) > 1 else None
    return {
        "first_event_s": first_event,
        "ttft_s": ttft,
        "total_s": events[-1][0] if events else None,
        "visible_tokens": len(text_times),
        "reasoning_tokens": n_reason,
        "tpot_ms": None if tpot is None else tpot * 1000,
        "decode_tps": None if not tpot else 1 / tpot,
    }


def from_ollama(d):
    ns = 1e9
    prefill_s = d["prompt_eval_duration"] / ns
    decode_s = d["eval_duration"] / ns
    return {
        "load_s": d.get("load_duration", 0) / ns,
        "prefill_s": prefill_s,
        "prefill_tps": d["prompt_eval_count"] / prefill_s if prefill_s > 0 else None,
        "decode_tps": d["eval_count"] / decode_s if decode_s > 0 else None,
        "cached_tokens": d.get("prompt_eval_cached_count", 0),
    }


def percentile(values, p):
    if not values:
        raise ValueError("percentile of empty list")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def weights_gb(params, bits_per_weight):
    return params * bits_per_weight / 8 / 1e9


def kv_cache_gib(layers, kv_heads, head_dim, tokens, users=1, bytes_per_value=2):
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens * users / 2**30


def summarize_runs(runs):
    ttfts = [r["ttft_s"] for r in runs if r["ttft_s"] is not None]
    rates = [r["decode_tps"] for r in runs if r["decode_tps"]]
    return {
        "runs": len(runs),
        "no_text": len(runs) - len(ttfts),
        "ttft_p50": percentile(ttfts, 50) if ttfts else None,
        "ttft_p95": percentile(ttfts, 95) if ttfts else None,
        "decode_tps_p50": statistics.median(rates) if rates else None,
    }


def stream_once(client, model, prompt=PROMPT, max_tokens=48):
    t0 = time.perf_counter()
    events = []
    stream = client.responses.create(model=model, input=prompt, stream=True, max_output_tokens=max_tokens, temperature=0)
    for ev in stream:
        now = time.perf_counter() - t0
        if ev.type == "response.output_text.delta":
            events.append((now, "text"))
        elif ev.type == "response.reasoning_summary_text.delta":
            events.append((now, "reasoning"))
        elif not events:
            events.append((now, "meta"))
    return summarize_events(events)


def server_decode_tps(model, prompt=PROMPT, max_tokens=48):
    import httpx

    url = BASE_URL.removesuffix("/v1") + "/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens, "temperature": 0}}
    return from_ollama(httpx.post(url, json=body, timeout=300).json())["decode_tps"]


def bench(models=MODELS, repeats=3):
    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key="ollama", timeout=300)
    report = {}
    for model in models:
        stream_once(client, model, max_tokens=1)  # làm nóng: nạp model trước khi đo
        report[model] = summarize_runs([stream_once(client, model) for _ in range(repeats)])
        report[model]["server_decode_tps"] = statistics.median(server_decode_tps(model) for _ in range(repeats))
    return report


if __name__ == "__main__":
    report = bench()
    num = lambda x, w, d: f"{'-':>{w}}" if x is None else f"{x:>{w}.{d}f}"
    print(f"{'model':<30}{'no text':>8}{'TTFT p50':>9}{'TTFT p95':>9}{'client t/s':>11}{'server t/s':>11}")
    for model, s in report.items():
        print(f"{model:<30}{s['no_text']:>8}{num(s['ttft_p50'], 9, 3)}{num(s['ttft_p95'], 9, 3)}"
              f"{num(s['decode_tps_p50'], 11, 1)}{num(s['server_decode_tps'], 11, 1)}")
    a, b = (report[m]["server_decode_tps"] for m in MODELS[:2])
    print(f"server decode ratio {MODELS[1]} / {MODELS[0]}: {b / a:.2f}x")
    print(f"weights 2.03B params at 16 / 8 / 4.5 bits: {weights_gb(2.03e9, 16):.2f} / {weights_gb(2.03e9, 8):.2f} / {weights_gb(2.03e9, 4.5):.2f} GB")
    print(f"qwen3:1.7b KV cache 4096 tokens x 8 users: {kv_cache_gib(28, 8, 128, 4096, users=8):.2f} GiB")
