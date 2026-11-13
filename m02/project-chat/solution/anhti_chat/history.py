def estimate_tokens(text: str) -> int:
    # ước lượng thô cho tiếng Việt với tokenizer của Qwen3: khoảng 3 ký tự một token, cộng phần khung của một tin nhắn
    return len(text) // 3 + 4


def build_context(messages: list[dict], summary_upto: int, budget: int, count=estimate_tokens) -> tuple[list[dict], list[dict]]:
    """Return (recent items to send, older messages that should be folded into the summary).

    Only whole turns are kept, newest first, and the last user message is always kept."""
    live = [m for m in messages if m["seq"] > summary_upto and m["status"] in ("complete", "replaced", "cancelled") and m["content"]]
    starts = [i for i, m in enumerate(live) if m["role"] == "user"]
    if not starts:
        return [], []
    keep_from, used = starts[-1], sum(count(m["content"]) for m in live[starts[-1]:])
    for s in reversed(starts[:-1]):
        cost = sum(count(m["content"]) for m in live[s:keep_from])
        if used + cost > budget:
            break
        keep_from, used = s, used + cost
    items = [{"role": m["role"], "content": m["content"]} for m in live[keep_from:]]
    return items, live[:keep_from]
