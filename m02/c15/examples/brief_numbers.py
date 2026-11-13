import json

import _project  # noqa: F401
import pandas as pd
from _project import PROJECT

from anhti_chat.config import DEMO_USERS

msgs = pd.read_csv(PROJECT.parents[0] / "data" / "messages.csv")
share = msgs["intent"].value_counts(normalize=True).round(3)
print("intent share in messages.csv (300 rows):", share.to_dict())
groups = {"bot answers with tools": ["order_status", "eta", "complaint_late"], "hand off to a human": ["refund", "change_address"],
          "bot answers without tools": ["other"]}
for name, intents in groups.items():
    print(f"  {name:<26} {share[intents].sum():.3f}")

orders = pd.read_csv(PROJECT / "data" / "orders.csv")
print("orders.csv:", len(orders), "orders,", orders["customer_id"].nunique(), "customers,", orders["status"].value_counts().to_dict())
for token, (cid, name) in DEMO_USERS.items():
    own = orders[orders["customer_id"] == cid]
    print(f"  {name:<5} {cid} {token:<11}", ", ".join(f"{o} {s}" for o, s in zip(own["order_id"], own["status"])))
snap = json.loads((PROJECT / "data" / "eta_snapshot.json").read_text())
print("P1 snapshot:", snap["model_version"], "test MAE", snap["test_mae"], "min,", len(snap["eta_minutes"]), "active orders")
