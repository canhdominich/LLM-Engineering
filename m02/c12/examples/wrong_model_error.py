from _setup import client

llm = client(max_retries=2)
llm.responses.create(model="qwen3:70b", input="Đơn AT047838 đâu?", max_output_tokens=20)  # model chưa được tải về
