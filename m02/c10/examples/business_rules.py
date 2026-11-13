from pydantic import ValidationError

from _rules import Extraction, error_text

BASE = {"status": "ok", "pickup_district": "Doi Thong", "dropoff_district": "Xom Vui", "items": 2,
        "cod_vnd": 2500000, "service": "standard", "missing": []}
CASES = {
    "as expected": {},
    "read 250k as 250": {"cod_vnd": 250},
    "above COD limit": {"cod_vnd": 25000000},
    "ok but pickup null": {"pickup_district": None},
    "same district": {"dropoff_district": "Doi Thong"},
}
for name, change in CASES.items():
    data = {**BASE, **change}
    try:
        Extraction.model_validate(data)
        print(f"{name:20} valid")
    except ValidationError as e:
        print(f"{name:20} {error_text(e)[:110]}")
