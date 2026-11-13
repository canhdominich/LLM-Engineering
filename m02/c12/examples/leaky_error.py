from common.llm import FakeLLM


def reply_text(llm, question):  # viết theo đối tượng trả về của SDK OpenAI
    r = llm.chat([{"role": "user", "content": question}])
    return r.output_text.strip()


print(reply_text(FakeLLM(["Đơn AT047838 đang giao."]), "Đơn AT047838 đâu?"))
