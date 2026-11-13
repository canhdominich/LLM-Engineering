from _shop import BASE_URL, MODEL, TOOLS, client, get_order_status

llm = client()
question = "Đơn AT047838 của tôi đang ở đâu?"
print("truth from database:", get_order_status("AT047838")["status"])

# Lần 1: không có công cụ, model chỉ có thể đoán
r1 = llm.responses.create(model=MODEL, input=question, reasoning={"effort": "none"}, max_output_tokens=160, temperature=0)
print("no tools      :", [o.type for o in r1.output], "|", r1.output_text.replace("**", "").strip()[:300])

# Lần 2: có công cụ nhưng không có chỉ dẫn hệ thống
r2 = llm.responses.create(model=MODEL, input=question, tools=TOOLS, reasoning={"effort": "none"}, max_output_tokens=120, temperature=0)
print("tools only    :", [o.type for o in r2.output], "|", (r2.output_text or "").strip()[:150])
for item in r2.output:
    if item.type == "function_call":
        print("  requested:", item.name, item.arguments, "(only a request, nothing ran)")
print("model:", MODEL, "via", BASE_URL)
