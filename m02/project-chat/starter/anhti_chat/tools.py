import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ORDER_ID = {"type": "string", "pattern": r"^AT\d{6}$", "description": "Order id: AT followed by 6 digits, for example AT047838"}

TOOLS = [
    {"type": "function", "name": "get_order_status", "strict": True,
     "description": "Return the current status (created, picked_up, out_for_delivery, delivered, cancelled) and district of one "
                    "order of the logged-in customer. Use it for 'where is my order'. It does not give the arrival time.",
     "parameters": {"type": "object", "properties": {"order_id": ORDER_ID}, "required": ["order_id"], "additionalProperties": False}},
    {"type": "function", "name": "predict_eta", "strict": True,
     "description": "Predict how many minutes are left until an order of the logged-in customer arrives. Use it for 'when will it "
                    "arrive' or 'how long'. Only for orders that are created, picked_up or out_for_delivery.",
     "parameters": {"type": "object", "properties": {"order_id": ORDER_ID}, "required": ["order_id"], "additionalProperties": False}},
]

ACTIVE = ("created", "picked_up", "out_for_delivery")
P1_FIELDS = ["created_at", "district", "distance_km", "items", "weight_kg", "order_value_vnd", "payment", "service", "vehicle",
             "shipper_exp_months", "is_rain"]


class OrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(pattern=r"^AT\d{6}$")


class EtaUnavailable(Exception):
    pass


class EtaService(Protocol):
    version: str

    def total_minutes(self, order: dict) -> float: ...


class HttpEta:
    """Calls POST /predict of the P1 service (module 01 project)."""

    def __init__(self, base_url: str = "", client: httpx.Client | None = None, timeout: float = 2.0):
        self.http = client or httpx.Client(base_url=base_url, timeout=timeout)
        self.version = "p1-http"

    def total_minutes(self, order):
        body = {k: order[k] for k in P1_FIELDS}
        body["created_at"] = pd.Timestamp(order["created_at"]).isoformat()
        try:
            r = self.http.post("/predict", json=body)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise EtaUnavailable(type(e).__name__) from e
        data = r.json()
        self.version = data["model_version"]
        return float(data["eta_minutes"])


class SnapshotEta:
    """P1 predictions frozen in a JSON file, so evals do not change when P1 is retrained."""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.version, self.minutes = data["model_version"], data["eta_minutes"]

    def total_minutes(self, order):
        if order["order_id"] not in self.minutes:
            raise EtaUnavailable("not in snapshot")
        return float(self.minutes[order["order_id"]])


class OrderBook:
    def __init__(self, path: Path):
        df = pd.read_csv(path, keep_default_na=False, dtype={"delivered_at": str})
        self.rows = {r["order_id"]: r for r in df.to_dict("records")}

    def owner(self, order_id: str) -> str | None:
        row = self.rows.get(order_id)
        return None if row is None else row["customer_id"]

    def owned_by(self, customer_id: str) -> list[str]:
        return [oid for oid, r in self.rows.items() if r["customer_id"] == customer_id]


def tool_error(code: str, message: str, retryable: bool = False) -> dict:
    return {"error": code, "message": message, "retryable": retryable}


class ToolRunner:
    def __init__(self, orders: OrderBook, eta: EtaService, now: datetime):
        self.orders, self.eta, self.now = orders, eta, now

    def _own(self, order_id: str, customer_id: str) -> dict | None:
        # TODO 14: trả về dòng đơn hàng chỉ khi đơn tồn tại và thuộc customer_id; mọi trường hợp khác trả None.
        return self.orders.rows.get(order_id)

    def get_order_status(self, order_id: str, customer_id: str) -> dict:
        row = self._own(order_id, customer_id)
        if row is None:
            return tool_error("order_not_found", f"No order {order_id} in this customer's account. Ask them to check the id.")
        out = {"order_id": order_id, "status": row["status"], "district": row["district"],
               "created_at": str(row["created_at"])[11:16]}
        if row["delivered_at"]:
            out["delivered_at"] = row["delivered_at"]
        return out  # không trả cột note: ghi chú do người gửi nhập là dữ liệu không đáng tin

    def predict_eta(self, order_id: str, customer_id: str) -> dict:
        row = self._own(order_id, customer_id)
        if row is None:
            return tool_error("order_not_found", f"No order {order_id} in this customer's account. Ask them to check the id.")
        if row["status"] not in ACTIVE:
            return tool_error("not_on_the_way", f"Order {order_id} is {row['status']}; there is no arrival time to predict.")
        try:
            total = self.eta.total_minutes(row)
        except EtaUnavailable as e:
            return tool_error("eta_unavailable", f"ETA service failed ({e}). Say the estimate is not available now.", True)
        elapsed = (self.now - pd.Timestamp(row["created_at"]).to_pydatetime()).total_seconds() / 60
        return {"order_id": order_id, "status": row["status"], "minutes_left": int(max(5, round(total - elapsed))),
                "model_version": self.eta.version}

    def run(self, name: str, arguments: str, customer_id: str) -> dict:
        funcs = {"get_order_status": self.get_order_status, "predict_eta": self.predict_eta}
        # TODO 15: tên lạ trả unknown_tool; kiểm arguments bằng OrderArgs (sai trả invalid_arguments);
        # chạy hàm với customer_id của phiên; ngoại lệ trả tool_failed, retryable=True.
        args = json.loads(arguments)
        return funcs[name](args["order_id"], customer_id)
