from _shop import INSTRUCTIONS, MODEL, TOOLS, client

llm = client()
question = "Đơn AT999999 của tôi đang ở đâu?"
call = {"type": "function_call", "call_id": "call_1", "name": "get_order_status", "arguments": '{"order_id": "AT999999"}'}
outputs = {
    "raw traceback": "Traceback (most recent call last): KeyError: 'AT999999'",
    "empty string": "",
    "structured error": '{"error": "order_not_found", "message": "No order AT999999. Ask the customer to check the id (AT + 6 digits).", "retryable": false}',
}
for label, text in outputs.items():
    # Lượt gọi công cụ được dựng sẵn để mọi phương án nhận đúng cùng một tình huống
    items = [{"role": "user", "content": question}, call, {"type": "function_call_output", "call_id": "call_1", "output": text}]
    r = llm.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=items, tools=TOOLS,
                             reasoning={"effort": "none"}, max_output_tokens=200, temperature=0)
    kinds = [o.type for o in r.output]
    print(f"{label:<17}{kinds}:", (r.output_text or "").strip()[:200])
print("model:", MODEL)
