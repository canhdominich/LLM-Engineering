import json
import time

from _shop import FUNCTIONS, INSTRUCTIONS, MODEL, TOOLS, client, dumps

llm = client()
question = "Hai đơn AT047838 và AT051122 của tôi đang ở đâu, và bao lâu nữa thì tới?"
items = [{"role": "user", "content": question}]
MAX_STEPS = 6
total_in = total_out = n_calls = 0
t_start = time.perf_counter()
for step in range(1, MAX_STEPS + 1):
    r = llm.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=items, tools=TOOLS,
                             reasoning={"effort": "none"}, max_output_tokens=400, temperature=0)
    total_in += r.usage.input_tokens
    total_out += r.usage.output_tokens
    calls = [o for o in r.output if o.type == "function_call"]
    items += [o.model_dump(exclude_none=True) for o in r.output]
    print(f"step {step}: input {r.usage.input_tokens} tokens, {len(calls)} calls in this response",
          [f"{c.name}({json.loads(c.arguments).get('order_id')})" for c in calls])
    if not calls:
        print("final answer:", r.output_text.strip())
        break
    for c in calls:
        n_calls += 1
        items.append({"type": "function_call_output", "call_id": c.call_id, "output": dumps(FUNCTIONS[c.name](**json.loads(c.arguments)))})
else:
    print("stopped: max_steps reached")
print(f"total: {step} model calls, {n_calls} tool calls, {total_in} input + {total_out} output tokens, {time.perf_counter() - t_start:.0f} s")
for oid in ("AT047838", "AT051122"):
    print("truth:", oid, FUNCTIONS["get_order_status"](oid)["status"], FUNCTIONS["predict_eta"](oid)["minutes_left"], "min")
