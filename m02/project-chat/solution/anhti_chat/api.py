import asyncio
import threading
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chat import ChatService, NotFound
from .config import DEMO_USERS

STATIC = Path(__file__).resolve().parent / "static"


class ChatIn(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=1000)


def create_app(service: ChatService, users: dict = DEMO_USERS) -> FastAPI:
    app = FastAPI(title="Giao Hang Anh Ti support chat")
    store = service.store

    def current_user(authorization: Annotated[str | None, Header()] = None) -> str:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token not in users:
            raise HTTPException(401, "invalid_token")
        return users[token][0]  # customer_id chỉ đến từ token, không bao giờ từ body hay từ model

    User = Annotated[str, Depends(current_user)]

    def owned(body: ChatIn, user: User) -> ChatIn:
        if body.conversation_id is not None and store.get_conversation(body.conversation_id, user) is None:
            raise HTTPException(404, "conversation_not_found")
        return body

    @app.post("/chat", response_class=EventSourceResponse)
    async def chat(body: Annotated[ChatIn, Depends(owned)], user: User, request: Request):
        cancel, lock = threading.Event(), threading.Lock()
        gen = service.handle(user, body.conversation_id, body.message, cancel)

        def step():
            with lock:
                return next(gen, None)

        def stop():
            cancel.set()  # vòng lặp trong thread dừng ở sự kiện kế tiếp của provider
            with lock:
                gen.close()  # lưu câu trả lời dở với status cancelled và đóng stream tới provider

        async def watch():
            while not await request.is_disconnected():
                await asyncio.sleep(0.2)
            await run_in_threadpool(stop)

        watcher = asyncio.create_task(watch())  # khi client ngắt, generator này bị treo ở yield nên cần một task riêng
        try:
            while (ev := await run_in_threadpool(step)) is not None:
                yield ServerSentEvent(data=ev.data, event=ev.name)
        finally:
            watcher.cancel()

    @app.post("/chat/complete")
    def chat_complete(body: Annotated[ChatIn, Depends(owned)], user: User) -> dict:
        try:
            events = list(service.handle(user, body.conversation_id, body.message))
        except NotFound:
            raise HTTPException(404, "conversation_not_found")
        text = "".join(e.data["text"] for e in events if e.name == "delta")
        for e in events:
            if e.name == "replace":
                text = e.data["text"]
        return {"conversation_id": events[0].data["conversation_id"], "answer": text,
                "events": [{"event": e.name, "data": e.data} for e in events if e.name != "delta"]}

    @app.get("/conversations")
    def conversations(user: User) -> list[dict]:
        return store.list_conversations(user)

    @app.get("/conversations/{cid}/messages")
    def messages(cid: str, user: User) -> list[dict]:
        if store.get_conversation(cid, user) is None:
            raise HTTPException(404, "conversation_not_found")  # 404 chứ không 403: không xác nhận hội thoại tồn tại
        return store.messages(cid)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "provider": service.llm.name, "model": service.llm.model}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def build_service(settings=None, llm=None, store=None, eta=None):
    from .config import load_settings
    from .llm import make_client
    from .store import PgStore
    from .tools import HttpEta, OrderBook, SnapshotEta, ToolRunner
    settings = settings or load_settings()
    llm = llm or make_client(settings)
    store = store or PgStore(settings.pg_dsn, settings.pg_schema).init()
    eta = eta or (HttpEta(settings.eta_url) if settings.eta_url else SnapshotEta(settings.eta_snapshot))
    return ChatService(llm, store, ToolRunner(OrderBook(settings.orders_path), eta, settings.clock), settings)
