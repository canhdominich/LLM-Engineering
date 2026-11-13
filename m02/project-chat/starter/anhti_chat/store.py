import json
import uuid
from datetime import datetime, timezone

import psycopg

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS {s};
CREATE TABLE IF NOT EXISTS {s}.conversations (
    id text PRIMARY KEY,
    user_id text NOT NULL,
    summary text NOT NULL DEFAULT '',
    summary_upto integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS conversations_user_idx ON {s}.conversations (user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS {s}.messages (
    conversation_id text NOT NULL REFERENCES {s}.conversations(id) ON DELETE CASCADE,
    seq integer NOT NULL,
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL,
    status text NOT NULL DEFAULT 'complete' CHECK (status IN ('complete', 'cancelled', 'error', 'replaced')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, seq));
CREATE TABLE IF NOT EXISTS {s}.usage_events (
    id bigserial PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES {s}.conversations(id) ON DELETE CASCADE,
    kind text NOT NULL CHECK (kind IN ('triage', 'chat', 'summary')),
    model text NOT NULL,
    input_tokens integer NOT NULL,
    output_tokens integer NOT NULL,
    latency_ms integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS {s}.tool_calls (
    id bigserial PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES {s}.conversations(id) ON DELETE CASCADE,
    name text NOT NULL,
    arguments text NOT NULL,
    outcome text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS {s}.handoffs (
    id bigserial PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES {s}.conversations(id) ON DELETE CASCADE,
    user_id text NOT NULL,
    intent text NOT NULL,
    reason text NOT NULL,
    order_id text,
    created_at timestamptz NOT NULL DEFAULT now());
"""


class PgStore:
    def __init__(self, dsn: str, schema: str = "m02_project"):
        self.dsn, self.s = dsn, schema

    def _conn(self):
        return psycopg.connect(self.dsn)

    def init(self) -> "PgStore":
        with self._conn() as conn:
            conn.execute(SCHEMA_SQL.format(s=self.s))
        return self

    def create_conversation(self, user_id: str) -> str:
        cid = uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute(f"INSERT INTO {self.s}.conversations (id, user_id) VALUES (%s, %s)", (cid, user_id))
        return cid

    def get_conversation(self, cid: str, user_id: str) -> dict | None:
        # TODO 1: trả về dict id, user_id, summary, summary_upto nếu hội thoại cid thuộc user_id, ngược lại None.
        # Đặt điều kiện user_id ngay trong câu truy vấn, đừng đọc ra rồi mới so trong Python.
        raise NotImplementedError("TODO 1")

    def list_conversations(self, user_id: str) -> list[dict]:
        # TODO 2: trả về danh sách {"id", "messages"} (số tin nhắn) các hội thoại của user_id, mới nhất trước.
        raise NotImplementedError("TODO 2")

    def add_message(self, cid: str, role: str, content: str, status: str = "complete") -> int:
        # TODO 3: khóa dòng hội thoại (SELECT ... FOR UPDATE), tính seq kế tiếp, chèn tin nhắn và trả về seq.
        raise NotImplementedError("TODO 3")

    def messages(self, cid: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(f"""SELECT seq, role, content, status FROM {self.s}.messages
                WHERE conversation_id = %s ORDER BY seq""", (cid,)).fetchall()
        return [{"seq": r[0], "role": r[1], "content": r[2], "status": r[3]} for r in rows]

    def set_summary(self, cid: str, summary: str, upto: int) -> None:
        with self._conn() as conn:
            conn.execute(f"UPDATE {self.s}.conversations SET summary = %s, summary_upto = %s WHERE id = %s", (summary, upto, cid))

    def add_usage(self, cid: str, kind: str, model: str, input_tokens: int, output_tokens: int, latency_ms: int) -> None:
        # TODO 4: ghi một dòng vào usage_events cho mỗi lời gọi model.
        raise NotImplementedError("TODO 4")

    def usage(self, cid: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(f"""SELECT kind, model, input_tokens, output_tokens, latency_ms FROM {self.s}.usage_events
                WHERE conversation_id = %s ORDER BY id""", (cid,)).fetchall()
        return [dict(zip(("kind", "model", "input_tokens", "output_tokens", "latency_ms"), r)) for r in rows]

    def add_tool_call(self, cid: str, name: str, arguments: str, outcome: str) -> None:
        with self._conn() as conn:
            conn.execute(f"INSERT INTO {self.s}.tool_calls (conversation_id, name, arguments, outcome) VALUES (%s, %s, %s, %s)",
                         (cid, name, arguments, outcome))

    def tool_calls(self, cid: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(f"SELECT name, arguments, outcome FROM {self.s}.tool_calls WHERE conversation_id = %s ORDER BY id",
                                (cid,)).fetchall()
        return [{"name": r[0], "arguments": r[1], "outcome": r[2]} for r in rows]

    def add_handoff(self, cid: str, user_id: str, intent: str, reason: str, order_id: str | None) -> int:
        with self._conn() as conn:
            return conn.execute(f"""INSERT INTO {self.s}.handoffs (conversation_id, user_id, intent, reason, order_id)
                VALUES (%s, %s, %s, %s, %s) RETURNING id""", (cid, user_id, intent, reason, order_id)).fetchone()[0]

    def handoffs(self, cid: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(f"SELECT id, intent, reason, order_id FROM {self.s}.handoffs WHERE conversation_id = %s ORDER BY id",
                                (cid,)).fetchall()
        return [{"id": r[0], "intent": r[1], "reason": r[2], "order_id": r[3]} for r in rows]

    def delete_conversation(self, cid: str) -> None:
        with self._conn() as conn:
            conn.execute(f"DELETE FROM {self.s}.conversations WHERE id = %s", (cid,))


class MemoryStore:
    """Same interface as PgStore, kept in a dict: for the eval runner and quick demos, not for production."""

    def __init__(self):
        self.convs, self.msgs, self.usage_rows, self.tools, self.hand = {}, {}, {}, {}, {}

    def init(self):
        return self

    def create_conversation(self, user_id):
        cid = uuid.uuid4().hex
        self.convs[cid] = {"id": cid, "user_id": user_id, "summary": "", "summary_upto": 0,
                           "created_at": datetime.now(timezone.utc)}
        for t in (self.msgs, self.usage_rows, self.tools, self.hand):
            t[cid] = []
        return cid

    def get_conversation(self, cid, user_id):
        c = self.convs.get(cid)
        return None if c is None or c["user_id"] != user_id else {k: c[k] for k in ("id", "user_id", "summary", "summary_upto")}

    def list_conversations(self, user_id):
        rows = sorted((c for c in self.convs.values() if c["user_id"] == user_id), key=lambda c: c["created_at"], reverse=True)
        return [{"id": c["id"], "messages": len(self.msgs[c["id"]])} for c in rows]

    def add_message(self, cid, role, content, status="complete"):
        seq = len(self.msgs[cid]) + 1
        self.msgs[cid].append({"seq": seq, "role": role, "content": content, "status": status})
        return seq

    def messages(self, cid):
        return [dict(m) for m in self.msgs.get(cid, [])]

    def set_summary(self, cid, summary, upto):
        self.convs[cid].update(summary=summary, summary_upto=upto)

    def add_usage(self, cid, kind, model, input_tokens, output_tokens, latency_ms):
        self.usage_rows[cid].append(dict(kind=kind, model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                                         latency_ms=latency_ms))

    def usage(self, cid):
        return list(self.usage_rows.get(cid, []))

    def add_tool_call(self, cid, name, arguments, outcome):
        self.tools[cid].append({"name": name, "arguments": arguments, "outcome": outcome})

    def tool_calls(self, cid):
        return list(self.tools.get(cid, []))

    def add_handoff(self, cid, user_id, intent, reason, order_id):
        hid = sum(len(v) for v in self.hand.values()) + 1
        self.hand[cid].append({"id": hid, "intent": intent, "reason": reason, "order_id": order_id})
        return hid

    def handoffs(self, cid):
        return list(self.hand.get(cid, []))

    def delete_conversation(self, cid):
        for t in (self.convs, self.msgs, self.usage_rows, self.tools, self.hand):
            t.pop(cid, None)
