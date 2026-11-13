from pydantic import BaseModel, ConfigDict, Field, ValidationError


class OrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(pattern=r"^AT\d{6}$")


# Những chuỗi arguments mà một model có thể trả về
candidates = [
    '{"order_id": "AT047838"}',
    '{"order_id": "47838"}',
    '{"order_id": "at047838"}',
    '{"order_id": "AT047838", "force": true}',
    '{"id": "AT047838"}',
]
for raw in candidates:
    try:
        args = OrderArgs.model_validate_json(raw)
        print(f"OK   {raw:<42} order_id={args.order_id}")
    except ValidationError as e:
        err = e.errors()[0]
        print(f"FAIL {raw:<42} {err['type']} at {err['loc']}")
