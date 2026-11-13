import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve()
LIVE_ORDERS = Path(os.environ.get("ANHTI_LIVE_ORDERS", HERE.parents[2] / "data" / "live_orders.csv"))
HISTORY = Path(os.environ.get("ANHTI_DATA", HERE.parents[4] / "m01" / "data")) / "orders.parquet"
NOW = pd.Timestamp("2026-11-05 10:30")
DISTRICTS = ["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]
SYSTEM = ("You are the support assistant of Giao Hang Anh Ti. Use the tools to look up facts and never guess an order status "
          "or an arrival time. Answer in Vietnamese in one or two short sentences.")
FALLBACK = "Sorry, I could not finish this request. A support agent will reply to you soon."

ORDER_ID_PARAM = {"type": "string", "pattern": r"^AT\d{6}$",
                  "description": "Order id: AT followed by 6 digits, for example AT047838"}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_status", "strict": True,
        "description": "Return where an order is now: created, picked_up, out_for_delivery, delivered or cancelled. "
                       "Use for questions like 'where is my order'. It does not give the arrival time.",
        "parameters": {"type": "object", "properties": {"order_id": ORDER_ID_PARAM}, "required": ["order_id"],
                       "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "predict_eta", "strict": True,
        "description": "Predict how many minutes are left until an order that is on the way arrives. "
                       "Use for 'when will it arrive' or 'how long'. Not for delivered or cancelled orders.",
        "parameters": {"type": "object", "properties": {"order_id": ORDER_ID_PARAM}, "required": ["order_id"],
                       "additionalProperties": False}}},
]


class OrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(pattern=r"^AT\d{6}$")


def tool_error(code, message, retryable=False):
    return {"error": code, "message": message, "retryable": retryable}


def load_orders(path=LIVE_ORDERS):
    df = pd.read_csv(path, parse_dates=["created_at"], keep_default_na=False)
    return df.set_index("order_id", drop=False)


def eta_features(df):
    hour = df["created_at"].dt.hour
    X = pd.DataFrame({
        "distance_km": df["distance_km"], "sqrt_items": np.sqrt(df["items"]), "rain": df["is_rain"].astype(float),
        "peak": hour.isin([11, 12, 17, 18, 19]).astype(float), "express": (df["service"] == "express").astype(float),
        "exp24": df["shipper_exp_months"].clip(upper=24), "weekend": (df["created_at"].dt.day_of_week >= 5).astype(float)})
    for d in DISTRICTS[1:]:
        X["d_" + d] = (df["district"] == d).astype(float)
    for v in ("van", "bike"):
        X["v_" + v] = (df["vehicle"] == v).astype(float)
    return X


def train_eta_model(path=HISTORY):
    hist = pd.read_parquet(path)
    return Ridge(alpha=1.0).fit(eta_features(hist), hist["delivery_minutes"])


def get_order_status(order_id, orders):
    if order_id not in orders.index:
        return tool_error("order_not_found", f"No order {order_id}. Ask the customer to check the id (AT + 6 digits).")
    row = orders.loc[order_id]
    return {"order_id": order_id, "status": row["status"], "district": row["district"]}


def predict_eta(order_id, orders, model, now=NOW):
    if order_id not in orders.index:
        return tool_error("order_not_found", f"No order {order_id}. Ask the customer to check the id (AT + 6 digits).")
    status = orders.loc[order_id, "status"]
    if status in ("delivered", "cancelled"):
        return tool_error("not_on_the_way", f"Order {order_id} is {status}; there is no arrival time to predict.")
    row = orders.loc[[order_id]]
    total = float(model.predict(eta_features(row))[0])
    elapsed = (now - row["created_at"].iloc[0]).total_seconds() / 60
    return {"order_id": order_id, "minutes_left": int(max(5, round(total - elapsed)))}


def make_registry(orders, model):
    return {"get_order_status": lambda order_id: get_order_status(order_id, orders),
            "predict_eta": lambda order_id: predict_eta(order_id, orders, model)}


def execute_call(name, arguments, registry):
    if name not in registry:
        return tool_error("unknown_tool", f"No tool named {name}. Available: {sorted(registry)}")
    try:
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
        args = OrderArgs.model_validate_json(raw)
    except ValidationError as e:
        return tool_error("invalid_arguments", e.errors()[0]["msg"])
    try:
        return registry[name](args.order_id)
    except Exception as e:  # lỗi bên trong công cụ trở thành dữ liệu cho model, không làm sập vòng lặp
        return tool_error("tool_failed", f"{type(e).__name__}: {e}", retryable=True)


@dataclass
class ChatResult:
    answer: str
    steps: int
    stopped: str
    tool_calls: list = field(default_factory=list)


def run_chat(llm, question, registry, max_steps=4):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    seen, log = set(), []
    for step in range(1, max_steps + 1):
        reply = llm.chat(messages, tools=TOOLS)
        if not reply.tool_calls:
            return ChatResult(reply.text, step, "answer", log)
        keys = [(c["name"], json.dumps(c["arguments"], sort_keys=True)) for c in reply.tool_calls]
        if any(k in seen for k in keys):
            return ChatResult(FALLBACK, step, "repeated_call", log)
        seen.update(keys)
        messages.append({"role": "assistant", "content": reply.text or "", "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"])}}
            for c in reply.tool_calls]})
        for c in reply.tool_calls:
            result = execute_call(c["name"], c["arguments"], registry)
            log.append({"name": c["name"], "arguments": c["arguments"], "result": result})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False)})
    return ChatResult(FALLBACK, max_steps, "max_steps", log)


def call(i, name, order_id):
    return {"id": f"call_{i}", "name": name, "arguments": {"order_id": order_id}}


def main():
    from common.llm import LLM, FakeLLM
    orders = load_orders()
    registry = make_registry(orders, train_eta_model())
    scenarios = [
        ("Don AT047838 dang o dau, bao lau nua toi?", FakeLLM([
            {"tool_calls": [call(1, "get_order_status", "AT047838"), call(2, "predict_eta", "AT047838")]},
            "Don AT047838 dang duoc giao o Cho Moi, khoang 11 phut nua toi."])),
        ("Don AT999999 dang o dau?", FakeLLM([
            {"tool_calls": [call(1, "get_order_status", "AT999999")]}, "Minh khong tim thay don nay, ban kiem tra lai ma don giup minh."])),
        ("Don AT039405 bao gio toi?", FakeLLM([
            {"tool_calls": [call(1, "predict_eta", "AT039405")]}, {"tool_calls": [call(2, "predict_eta", "AT039405")]}])),
    ]
    if os.environ.get("LLM_PROVIDER") == "ollama":
        scenarios.append(("Đơn AT047838 đang ở đâu, bao lâu nữa tới?", LLM()))
    for question, llm in scenarios:
        result = run_chat(llm, question, registry)
        print(f"[{llm.model}] {question}")
        for c in result.tool_calls:
            print(f"  {c['name']}({c['arguments']['order_id'] if isinstance(c['arguments'], dict) else c['arguments']}):",
                  json.dumps(c["result"], ensure_ascii=False)[:90])
        print(f"  stopped={result.stopped} steps={result.steps} answer={result.answer}")


if __name__ == "__main__":
    main()
