import math
import os
import statistics
import time

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
MODELS = os.environ.get("BENCH_MODELS", "smollm2:135m-instruct-q8_0,smollm2:135m-instruct-q4_K_M").split(",")
PROMPT = "A customer asks: my parcel AT027075 is late, what should I do? Answer in two short sentences."


def summarize_events(events):
    # TODO 1: events là list (giây, loại) với loại "meta", "reasoning" hoặc "text".
    # Trả dict: first_event_s, ttft_s (thời điểm "text" đầu tiên), total_s, visible_tokens,
    # reasoning_tokens, tpot_ms, decode_tps. Không có "text" thì ttft_s là None; dưới 2 token "text" thì tpot_ms, decode_tps là None.
    raise NotImplementedError

def from_ollama(d):
    # TODO 2: đổi các trường *_duration (nano giây) của /api/generate thành giây;
    # trả load_s, prefill_s, prefill_tps, decode_tps, cached_tokens (prompt_eval_cached_count, mặc định 0).
    raise NotImplementedError

def percentile(values, p):
    # TODO 3: phân vị p (0 đến 100) có nội suy tuyến tính như numpy.percentile; không sửa list đầu vào;
    # list rỗng hoặc p ngoài [0, 100] thì báo ValueError.
    raise NotImplementedError

def weights_gb(params, bits_per_weight):
    # TODO 4a: dung lượng trọng số theo GB (1e9 byte).
    raise NotImplementedError


def kv_cache_gib(layers, kv_heads, head_dim, tokens, users=1, bytes_per_value=2):
    # TODO 4b: bộ nhớ KV cache theo GiB (2**30 byte), nhớ cả key lẫn value.
    raise NotImplementedError

def summarize_runs(runs):
    # TODO 5: gộp nhiều lần chạy: runs, no_text (số lần không có chữ), ttft_p50, ttft_p95,
    # decode_tps_p50 (trung vị). Bỏ qua giá trị None khi tính.
    raise NotImplementedError

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
