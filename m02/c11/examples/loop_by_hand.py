import json

from _shop import FUNCTIONS, dumps

# Hai phản hồi model được ghi sẵn theo đúng dạng item của Responses API
scripted = [
    [{"type": "function_call", "call_id": "call_1", "name": "get_order_status", "arguments": '{"order_id": "AT047838"}'}],
    [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Đơn AT047838 đang được giao ở Cho Moi."}]}],
]

items = [{"role": "user", "content": "Đơn AT047838 của tôi đang ở đâu?"}]
for turn, output in enumerate(scripted, start=1):
    items += output  # lưu lại đúng những gì model trả về
    calls = [x for x in output if x["type"] == "function_call"]
    print(f"turn {turn}: model returned {[x['type'] for x in output]}")
    if not calls:
        print("final answer:", output[0]["content"][0]["text"])
        break
    for call in calls:
        args = json.loads(call["arguments"])  # arguments là một chuỗi JSON, không phải dict
        result = FUNCTIONS[call["name"]](**args)  # chính code của ta chạy hàm
        items.append({"type": "function_call_output", "call_id": call["call_id"], "output": dumps(result)})
        print(f"  code ran {call['name']}({args}) returned {dumps(result)}")
print("items in the conversation:", len(items))
for x in items:
    print("  ", x.get("type", "message(" + x.get("role", "") + ")"))
