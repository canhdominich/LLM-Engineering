from _setup import INSTRUCTIONS, MODEL, Timer, client

llm = client()
QUESTION = "Phí giao hàng nội thành Pho Rom Ra tính thế nào?"


def ask(limit, reasoning):
    kw = {"reasoning": {"effort": "none"}} if not reasoning else {}
    with Timer() as t:
        r = llm.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=QUESTION,
                                 max_output_tokens=limit, temperature=0, **kw)
    text = r.output_text.strip()
    hit_limit = r.usage.output_tokens >= limit  # Ollama không báo incomplete, tự kiểm
    print(f"thinking={'on ' if reasoning else 'off'} limit={limit:<4} status={r.status:<9} "
          f"in={r.usage.input_tokens} out={r.usage.output_tokens:<4} hit_limit={hit_limit!s:<5} "
          f"answer_chars={len(text):<4} {t.seconds:.1f} s")
    print("   items:", [o.type for o in r.output], "| answer:", (text[:90] or "<empty>"))


ask(60, reasoning=True)
ask(700, reasoning=True)
