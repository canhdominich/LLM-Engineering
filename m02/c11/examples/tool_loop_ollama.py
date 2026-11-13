import json
import time

from _shop import FUNCTIONS, INSTRUCTIONS, MODEL, TOOLS, client, dumps

llm = client()
items = [{"role": "user", "content": "Đơn AT047838 của tôi đang ở đâu?"}]
for turn in range(1, 5):
    t0 = time.perf_counter()
    r = llm.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=items, tools=TOOLS,
                             reasoning={"effort": "none"}, max_output_tokens=300, temperature=0)
    sec = time.perf_counter() - t0
    calls = [o for o in r.output if o.type == "function_call"]
    print(f"turn {turn}: {sec:.1f} s, input {r.usage.input_tokens} tokens, output {r.usage.output_tokens} tokens")
    items += [o.model_dump(exclude_none=True) for o in r.output]
    if not calls:
        print("final answer:", r.output_text.strip())
        break
    for call in calls:
        result = FUNCTIONS[call.name](**json.loads(call.arguments))
        items.append({"type": "function_call_output", "call_id": call.call_id, "output": dumps(result)})
        print(f"  model asked {call.name}{call.arguments}; code returned {dumps(result)}")
print("model:", r.model)
