import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

LAB = Path(__file__).resolve().parents[3]
LIVE = LAB / "m02" / "c11" / "data" / "live_orders.csv"
HISTORY = LAB / "m01" / "data" / "orders.parquet"
NOW = pd.Timestamp("2026-11-05 10:30")
ORDER_ID = re.compile(r"^AT\d{6}$")
DISTRICTS = ["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]
MODEL = "qwen3:1.7b"
BASE_URL = "http://localhost:11434/v1"


def load_orders():
    df = pd.read_csv(LIVE, parse_dates=["created_at"], keep_default_na=False)
    return df.set_index("order_id", drop=False)


def features(df):
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


@lru_cache(maxsize=1)
def eta_model():
    hist = pd.read_parquet(HISTORY)
    return Ridge(alpha=1.0).fit(features(hist), hist["delivery_minutes"])


def get_order_status(order_id):
    orders = load_orders()
    if order_id not in orders.index:
        raise KeyError(order_id)
    row = orders.loc[order_id]
    return {"order_id": order_id, "status": row["status"], "district": row["district"], "note": row["note"]}


def predict_eta(order_id):
    orders = load_orders()
    if order_id not in orders.index:
        raise KeyError(order_id)
    row = orders.loc[[order_id]]
    total = float(eta_model().predict(features(row))[0])
    elapsed = (NOW - row["created_at"].iloc[0]).total_seconds() / 60
    return {"order_id": order_id, "minutes_left": max(5, round(total - elapsed)), "model": "ridge_m01"}


def fn_tool(name, description, properties, required=None):
    return {"type": "function", "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": properties, "required": required or list(properties),
                           "additionalProperties": False}}


ORDER_ARG = {"order_id": {"type": "string", "description": "Order id: AT followed by 6 digits, for example AT047838"}}
TOOLS = [
    fn_tool("get_order_status", "Return where an order is now: created, picked_up, out_for_delivery, delivered or cancelled. "
            "Use for questions like 'where is my order'. It does not give the arrival time.", ORDER_ARG),
    fn_tool("predict_eta", "Predict how many minutes are left until an order that is on the way arrives. "
            "Use for 'when will it arrive' or 'how long'.", ORDER_ARG),
]
FUNCTIONS = {"get_order_status": get_order_status, "predict_eta": predict_eta}
INSTRUCTIONS = ("You are the support assistant of Giao Hang Anh Ti. Use the tools to look up facts and never guess an order status "
                "or an arrival time. Answer in Vietnamese in one or two short sentences.")


def client():
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key="ollama", timeout=600)


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)
