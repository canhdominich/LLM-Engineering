from transformers import AutoTokenizer

from _chat_log import TURNS, messages_until

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")  # chỉ tải tokenizer của đúng model đang dùng


def count(msgs):
    return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"])


total = 0
for n in range(1, len(TURNS) + 1):
    msgs = messages_until(n)
    if msgs[-1]["role"] == "assistant":
        msgs = msgs[:-1]  # lượt n: gửi lịch sử cũ cộng câu hỏi mới, chưa có câu trả lời
    sent = count(msgs)
    total += sent
    if n in (1, 2, 5, 10):
        print(f"turn {n:>2}: messages sent {len(msgs):>2}, input tokens {sent:>4}, cumulative input {total:>5}")
first, last = count(messages_until(1)[:-1]), count(messages_until(10))
grow = (last - first) / 9  # mỗi lượt thêm khoảng chừng này token vào mọi lời gọi sau
print(f"turn 10 sends {last / first:.1f}x the tokens of turn 1; each turn adds about {grow:.0f} tokens")
for turns in (10, 20, 40):
    est = sum(first + grow * (k - 1) for k in range(1, turns + 1))
    print(f"estimate for {turns} turns: about {est:,.0f} cumulative input tokens")
