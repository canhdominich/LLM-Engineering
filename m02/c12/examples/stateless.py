from _setup import INSTRUCTIONS, MODEL, client

llm = client()
TURN1 = "Chào bạn, tôi là Lan. Đơn của tôi là AT047838, giao tới Ben Song."
TURN2 = "Đơn của tôi mã là gì và giao tới quận nào?"


def send(input_items):
    r = llm.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=input_items,
                             max_output_tokens=80, temperature=0, reasoning={"effort": "none"})
    return r.output_text.strip(), r.usage.input_tokens


answer1, tok1 = send([{"role": "user", "content": TURN1}])
print(f"turn 1 (input {tok1} tokens): {answer1}")
answer2, tok2 = send([{"role": "user", "content": TURN2}])  # lời gọi mới, không gửi lại lượt 1
print(f"turn 2 without history (input {tok2} tokens): {answer2}")
history = [{"role": "user", "content": TURN1}, {"role": "assistant", "content": answer1},
           {"role": "user", "content": TURN2}]
answer3, tok3 = send(history)
print(f"turn 2 with history (input {tok3} tokens): {answer3}")
print("mentions AT047838 without / with history:", "AT047838" in answer2, "/", "AT047838" in answer3)
