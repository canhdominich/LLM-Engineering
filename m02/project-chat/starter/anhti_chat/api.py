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
        # TODO 6: đọc token sau "Bearer ", trả 401 nếu không có trong users, ngược lại trả customer_id.
        raise NotImplementedError("TODO 6")

    User = Annotated[str, Depends(current_user)]

    def owned(body: ChatIn, user: User) -> ChatIn:
        # TODO 7: nếu body.conversation_id không thuộc user thì trả 404 conversation_not_found.
        return body

    @app.post("/chat", response_class=EventSourceResponse)
    async def chat(body: Annotated[ChatIn, Depends(owned)], user: User, request: Request):
        # TODO 11: chạy service.handle trong threadpool, yield ServerSentEvent(data=ev.data, event=ev.name) cho từng sự kiện.
        # Khi client ngắt kết nối (request.is_disconnected()), bật cờ cancel và đóng generator để lưu câu trả lời dở.
        raise NotImplementedError("TODO 11")
        yield

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
        # TODO 7: cùng kiểm tra quyền sở hữu như trên trước khi trả tin nhắn.
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
