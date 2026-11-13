from _setup import INSTRUCTIONS, MODEL, Timer, client

llm = client()
with Timer() as t:
    r = llm.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input="Phí giao hàng nội thành Pho Rom Ra tính thế nào?",
        max_output_tokens=120,
        temperature=0,
        reasoning={"effort": "none"},  # tắt suy nghĩ của qwen3 để tiết kiệm token
    )
print("status:", r.status)
print("model:", r.model)
print("answer:", r.output_text.strip())
u = r.usage
print(f"usage: input {u.input_tokens}, output {u.output_tokens}, total {u.total_tokens} tokens")
print(f"output items: {[o.type for o in r.output]}")
print(f"seconds: {t.seconds:.1f}")
