import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "messages.csv"
LABELS = ["order_status", "eta", "complaint_late", "refund", "change_address", "other"]
LABEL_RE = re.compile(r"\b(" + "|".join(LABELS) + r")\b")


def messages():
    return pd.read_csv(DATA)


def sample24():
    # mỗi ý định lấy 4 tin từ 4 template khác nhau của tập test
    df = messages()
    test = df[df["split"] == "test"].sort_values("message_id")
    firsts = test.drop_duplicates("template_id")
    return firsts.groupby("intent", sort=False).head(4).sort_values("message_id").reset_index(drop=True)


def parse_label(text):
    found = set(LABEL_RE.findall(text.lower()))
    return found.pop() if len(found) == 1 else "invalid"


def sample12():
    return sample24().groupby("intent", sort=False).tail(2).sort_values("message_id").reset_index(drop=True)
