from _chat_log import messages_until
from _setup import MODEL, Timer, client

llm = client()
old = messages_until(6)[1:]  # sáu lượt đầu, bỏ system prompt
transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
with Timer() as t:
    r = llm.responses.create(
        model=MODEL, temperature=0, max_output_tokens=120, reasoning={"effort": "none"},
        instructions="Tóm tắt hội thoại hỗ trợ thành một câu ghi nhớ. Giữ nguyên mã đơn, tên người, quận, số điện thoại.",
        input=transcript)
summary = r.output_text.strip()
print("summary:", summary)
print(f"tokens: transcript in {r.usage.input_tokens}, summary out {r.usage.output_tokens}, {t.seconds:.1f} s")
for fact in ("AT047838", "Ben Song", "Minh", "0901 234 567"):
    print(f"   keeps {fact!r}: {fact in summary}")
