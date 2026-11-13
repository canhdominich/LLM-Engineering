import time

import _project  # noqa: F401
import pandas as pd
from _project import PROJECT

from anhti_chat.config import load_settings
from anhti_chat.llm import make_client
from anhti_chat.triage import classify, handoff_reason

client = make_client(load_settings(provider="ollama"))
msgs = pd.read_csv(PROJECT.parents[0] / "data" / "messages.csv").set_index("message_id")
for mid in ["M0004", "M0001", "M0003", "M0002"]:
    row = msgs.loc[mid]
    t0 = time.perf_counter()
    t, attempts, usages = classify(client, "", [{"role": "user", "content": row["text"]}])
    print(f"{mid} label={row['intent']:<14} got={t.intent:<14} order_id={t.order_id} needs_human={t.needs_human} "
          f"attempts={attempts} tokens={sum(u.input_tokens + u.output_tokens for u in usages)} {time.perf_counter() - t0:.0f} s")
    print(f"      text={row['text']!r} handoff={handoff_reason(t)}")
