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

# TODO 1: khai báo hai công cụ theo định dạng Chat Completions: {"type": "function", "function": {...}}.
# Mỗi công cụ: name, "strict": True, description từ 60 ký tự trở lên nói rõ khi nào dùng và khi nào không,
# parameters là object có duy nhất order_id (dùng ORDER_ID_PARAM), required ["order_id"], additionalProperties False.
TOOLS = []


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
    # TODO 2: trả về {"order_id", "status", "district"} của đơn; đơn không có trong orders thì trả về
    # tool_error("order_not_found", ...) thay vì ném KeyError.
    raise NotImplementedError


def predict_eta(order_id, orders, model, now=NOW):
    # TODO 3: đơn không tồn tại trả về order_not_found; đơn delivered hoặc cancelled trả về not_on_the_way.
    # Còn lại: tổng số phút dự đoán bằng model.predict(eta_features(...)) trừ số phút đã trôi qua từ created_at tới now,
    # tối thiểu 5, làm tròn thành int: {"order_id": ..., "minutes_left": ...}.
    raise NotImplementedError


def make_registry(orders, model):
    return {"get_order_status": lambda order_id: get_order_status(order_id, orders),
            "predict_eta": lambda order_id: predict_eta(order_id, orders, model)}


def execute_call(name, arguments, registry):
    # TODO 4: không bao giờ ném ngoại lệ. Tên lạ: unknown_tool. arguments (dict hoặc chuỗi JSON) không qua OrderArgs:
    # invalid_arguments. Công cụ ném ngoại lệ: tool_failed với retryable=True và tên ngoại lệ trong message.
    raise NotImplementedError


@dataclass
class ChatResult:
    answer: str
    steps: int
    stopped: str
    tool_calls: list = field(default_factory=list)


def run_chat(llm, question, registry, max_steps=4):
    # TODO 5: vòng lặp tool calling có giới hạn.
    # messages bắt đầu bằng system (SYSTEM) và user. Mỗi bước gọi llm.chat(messages, tools=TOOLS).
    # Không có tool_calls: trả về ChatResult(reply.text, step, "answer", log).
    # Có lời gọi đã thấy (cùng name và arguments): trả về ChatResult(FALLBACK, step, "repeated_call", log).
    # Ngược lại: thêm một message assistant có tool_calls (id, type "function", function.name, function.arguments là chuỗi JSON),
    # rồi mỗi lời gọi một message role "tool" với tool_call_id và content là JSON của execute_call, đúng thứ tự.
    # Hết max_steps: trả về ChatResult(FALLBACK, max_steps, "max_steps", log).
    raise NotImplementedError


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
