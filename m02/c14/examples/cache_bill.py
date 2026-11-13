from statistics import mean

import tiktoken

from _common import PROMPTS, few_shot_prompt, golden, prices, wrap

enc = tiktoken.get_encoding("o200k_base")  # gần đúng cho model OpenAI; tokenizer gpt-6 chưa có trong tiktoken
api = prices()["api"]
REST = round(mean(len(enc.encode(f"Request M0001 received at 2026-11-05 09:00:00.\n{wrap(t)}")) for t in golden().text))
OUT = 3


def bill(n, prefix, p, mode):
    usd = lambda tok, kind: tok * p[kind] / 1e6
    if mode == "no cache" or prefix + REST < api["min_cacheable_tokens"]:
        return n * (usd(prefix + REST, "input") + usd(OUT, "output"))
    if mode == "stamp first":  # phần đầu không bao giờ khớp: lần nào cũng ghi cache mới
        return n * (usd(prefix + REST, "cache_write_30m") + usd(OUT, "output"))
    if mode == "stable first":  # breakpoint ngầm ở cuối tin của người dùng
        first, later = usd(prefix + REST, "cache_write_30m"), usd(prefix, "cached_input") + usd(REST, "cache_write_30m")
    else:  # breakpoint tường minh ngay sau phần cố định
        first, later = usd(prefix, "cache_write_30m") + usd(REST, "input"), usd(prefix, "cached_input") + usd(REST, "input")
    return first + (n - 1) * later + n * usd(OUT, "output")


print(f"prices checked on {api['checked_on']} at {api['source']}")
for name, text in [("full prompt", PROMPTS["full"]), ("36-shot prompt", few_shot_prompt(6))]:
    prefix = len(enc.encode(text))
    print(f"{name}: prefix {prefix} tokens + {REST} changing tokens + {OUT} output tokens, 1,000 calls in 30 minutes")
    for model in ("gpt-6-luna", "gpt-6.1-sol"):
        p = api["models"][model]
        cells = [f"{m} ${bill(1000, prefix, p, m):.3f}" for m in ("no cache", "stamp first", "stable first", "explicit")]
        print(f"  {model:12} " + " | ".join(cells))
