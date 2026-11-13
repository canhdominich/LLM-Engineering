from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from _fake_stream import FakeStreamLLM

ANSWER = "Đơn AT047838 đang được giao ở Cho Moi, shipper dự kiến tới sau khoảng 11 phút nữa."
llm = FakeStreamLLM(ANSWER)
app = FastAPI()


class ChatIn(BaseModel):
    message: str


@app.post("/chat", response_class=EventSourceResponse)
async def chat(body: ChatIn):
    n = 0
    async for delta in llm.stream([{"role": "user", "content": body.message}]):
        n += 1
        yield ServerSentEvent(data={"text": delta}, event="delta")  # gửi ngay từng mẩu
    yield ServerSentEvent(data={"chunks": n}, event="done")


@app.post("/chat-buffered", response_class=EventSourceResponse)
async def chat_buffered(body: ChatIn):
    parts = [d async for d in llm.stream([{"role": "user", "content": body.message}])]  # gom hết rồi mới gửi
    yield ServerSentEvent(data={"text": "".join(parts)}, event="delta")
    yield ServerSentEvent(data={"chunks": len(parts)}, event="done")
