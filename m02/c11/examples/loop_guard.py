import json

FIRST_INPUT, PER_ROUND = 334, 72  # số token đo ở bài 1: lượt đầu 334, mỗi vòng có 1 lời gọi thêm khoảng 72


def stuck_model(items):
    # Model "kẹt": luôn xin lại cùng một lời gọi, không bao giờ trả lời
    return {"type": "function_call", "name": "get_order_status", "arguments": '{"order_id": "AT047838"}'}


def run_loop(max_steps, stop_on_repeat):
    items, seen, total_in = ["user question"], set(), 0
    for step in range(1, max_steps + 1):
        total_in += FIRST_INPUT + PER_ROUND * (step - 1)  # mỗi vòng gửi lại toàn bộ lịch sử
        call = stuck_model(items)
        key = (call["name"], json.dumps(json.loads(call["arguments"]), sort_keys=True))
        if stop_on_repeat and key in seen:
            return step, "repeated_call", total_in
        seen.add(key)
        items += [call, "tool result"]
    return max_steps, "max_steps", total_in


for label, max_steps, repeat in [("no real limit (cap 20)", 20, False), ("max_steps=4", 4, False), ("max_steps=4 + repeat check", 4, True)]:
    steps, reason, tokens = run_loop(max_steps, repeat)
    print(f"{label:<28} stopped after {steps:>2} model calls, reason={reason:<13} input tokens={tokens:>6}")
