import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab" / "solution"))
from intent_lora import DATA, LABELS, normalize_text

SYSTEM = "Phân loại ý định tin nhắn khách hàng của Giao Hàng Anh Tí. Trả về đúng một nhãn: " + ", ".join(LABELS)

df = pd.read_csv(DATA)
train = df[df.split == "train"]
records, seen = [], set()
for text, intent in zip(train.text, train.intent):
    key = normalize_text(text)
    if key in seen:
        continue
    seen.add(key)
    records.append({"messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": text},
                                 {"role": "assistant", "content": intent}]})

print(json.dumps(records[0], ensure_ascii=False)[:230] + "...")
print(f"train rows: {len(train)}, kept after dedup: {len(records)}")
print("labels:", dict(sorted(Counter(r["messages"][2]["content"] for r in records).items())))
test_keys = {normalize_text(t) for t in df[df.split == "test"].text}
print(f"exact duplicates between train and test after normalizing: {len(seen & test_keys)}")
