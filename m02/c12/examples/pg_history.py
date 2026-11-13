import psycopg

from _chat_log import TURNS
from _setup import DSN, SCHEMA

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};
CREATE TABLE IF NOT EXISTS {SCHEMA}.demo_messages (
    conversation_id text NOT NULL,
    seq integer NOT NULL,
    role text NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, seq)
)"""

with psycopg.connect(DSN) as conn:
    conn.execute(DDL)
    conn.execute(f"DELETE FROM {SCHEMA}.demo_messages WHERE conversation_id = %s", ("conv-lan-1",))
    seq = 0
    for user, bot in TURNS[:3]:
        for role, content in (("user", user), ("assistant", bot)):
            seq += 1
            conn.execute(f"INSERT INTO {SCHEMA}.demo_messages (conversation_id, seq, role, content) VALUES (%s, %s, %s, %s)",
                         ("conv-lan-1", seq, role, content))  # tham số %s, không ghép chuỗi
    conn.commit()

with psycopg.connect(DSN) as conn:  # kết nối mới, như một request khác hoặc sau khi khởi động lại
    rows = conn.execute(f"SELECT seq, role, content FROM {SCHEMA}.demo_messages WHERE conversation_id = %s ORDER BY seq",
                        ("conv-lan-1",)).fetchall()
history = [{"role": role, "content": content} for _, role, content in rows]
print(f"loaded {len(history)} messages for conv-lan-1")
for seq, role, content in rows:
    print(f"  {seq} {role:<9} {content[:60]}")
try:
    with psycopg.connect(DSN) as conn:
        conn.execute(f"INSERT INTO {SCHEMA}.demo_messages (conversation_id, seq, role, content) VALUES (%s, %s, %s, %s)",
                     ("conv-lan-1", 3, "user", "duplicate seq"))
except psycopg.errors.UniqueViolation as e:
    print("second writer with seq 3:", type(e).__name__)
