from transformers import AutoTokenizer

from _chat_log import messages_until

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def count(msgs):
    return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"])


def trim(msgs, budget, pinned=None):
    system, rest = msgs[0], msgs[1:]
    head = [system] + ([{"role": "system", "content": pinned}] if pinned else [])
    starts = [i for i, m in enumerate(rest) if m["role"] == "user"]  # chỉ cắt ở đầu một lượt
    for s in starts:
        kept = head + rest[s:]
        if count(kept) <= budget:
            return kept, s // 2
    return head + rest[starts[-1]:], len(starts) - 1


full = messages_until(10)
print(f"full history: {len(full)} messages, {count(full)} tokens")
kept, dropped = trim(full, budget=250)
text = " ".join(m["content"] for m in kept)
print(f"trim to 250: {len(kept)} messages, {count(kept)} tokens, dropped first {dropped} turns")
print("   first kept user turn:", kept[1]["content"])
print("   still knows Ben Song / AT047838:", "Ben Song" in text, "/", "AT047838" in text)
facts = "Ghi nhớ: khách Lan, đơn AT047838, giao tới Ben Song, người nhận đổi thành Minh (0901 234 567)."
kept, dropped = trim(full, budget=250, pinned=facts)
text = " ".join(m["content"] for m in kept)
print(f"trim to 250 + pinned facts: {len(kept)} messages, {count(kept)} tokens, dropped first {dropped} turns")
print("   still knows Ben Song / AT047838:", "Ben Song" in text, "/", "AT047838" in text)
