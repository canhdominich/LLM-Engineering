import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
df = pd.read_csv(DATA)

strict = re.compile(r"\bAT\d{6}\b")
relaxed = re.compile(r"\bAT\d{6}\b", re.IGNORECASE)  # khách gõ không dấu thường viết thường cả mã đơn

has_id = df["text"].str.contains(relaxed)
print(f"messages: {len(df)}")
print(f"with an id, strict pattern (upper case only): {df['text'].str.contains(strict).sum()}")
print(f"with an id, relaxed pattern (any case)      : {has_id.sum()}")
for text in df.loc[has_id, "text"].head(3):
    found = relaxed.search(text).group(0).upper()
    print(f"  {found} found in {text[:46]!r}")
