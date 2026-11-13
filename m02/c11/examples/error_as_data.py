import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from _shop import FUNCTIONS, dumps, load_orders


class OrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(pattern=r"^AT\d{6}$")


def error(code, message, retryable=False):
    return {"error": code, "message": message, "retryable": retryable}


def safe_execute(name, raw_arguments):
    if name not in FUNCTIONS:
        return error("unknown_tool", f"No tool named {name}. Available: {sorted(FUNCTIONS)}")
    try:
        args = OrderArgs.model_validate_json(raw_arguments)
    except ValidationError as e:
        return error("invalid_arguments", e.errors()[0]["msg"])
    orders = load_orders()
    if args.order_id not in orders.index:
        return error("order_not_found", f"No order {args.order_id}. Ask the customer to check the id (AT + 6 digits).")
    status = orders.loc[args.order_id, "status"]
    if name == "predict_eta" and status in ("delivered", "cancelled"):
        return error("not_on_the_way", f"Order {args.order_id} is {status}; there is no arrival time to predict.")
    return FUNCTIONS[name](args.order_id)


print("naive predict_eta on a delivered order:", FUNCTIONS["predict_eta"]("AT039405"))
cases = [("get_order_status", '{"order_id": "AT999999"}'), ("get_order_status", '{"order_id": "47838"}'),
         ("cancel_order", '{"order_id": "AT047838"}'), ("predict_eta", '{"order_id": "AT039405"}'),
         ("predict_eta", '{"order_id": "AT047838"}')]
for name, raw in cases:
    out = safe_execute(name, raw)
    print(f"{name:<17}{json.loads(raw)['order_id']:<9} {out.get('error', 'ok'):<17}", out.get("message", dumps(out))[:72])
