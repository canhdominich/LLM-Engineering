import json

from pydantic import ValidationError

from _delivery import DeliveryRequest

schema = DeliveryRequest.model_json_schema()
print("required:", schema["required"])
print("additionalProperties:", schema["additionalProperties"])
for name, prop in schema["properties"].items():
    rules = {k: v for k, v in prop.items() if k not in ("title", "description")}
    print(f"  {name}: {json.dumps(rules)}")

good = '{"pickup_district": "Bai Bien", "dropoff_district": "Dau Pho", "items": 4, "cod_vnd": 2000000, "service": "standard"}'
print(DeliveryRequest.model_validate_json(good))

# Cùng schema, phía code: bắt lỗi kiểu và giá trị trước khi dữ liệu vào hệ thống
bad = '{"pickup_district": "Quận Bãi Biển", "dropoff_district": "Dau Pho", "items": "bốn", "cod_vnd": "2000 nghìn", "service": "standard", "note": "nha"}'
try:
    DeliveryRequest.model_validate_json(bad)
except ValidationError as e:
    print(f"{e.error_count()} errors")
    for err in e.errors():
        print(f"  {err['loc'][0]}: {err['type']}")
