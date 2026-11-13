from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

District = Literal["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]
FIELDS = ["pickup_district", "dropoff_district", "items", "cod_vnd", "service"]
COD_LIMIT = 5_000_000


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "need_info", "not_request"] = Field(
        description="ok: a delivery request with every field; need_info: a delivery request missing some field; not_request: anything else")
    pickup_district: Optional[District]
    dropoff_district: Optional[District]
    items: Optional[int] = Field(ge=1, le=30)
    cod_vnd: Optional[int] = Field(ge=0, description="Cash to collect in VND, 0 if prepaid. 250k means 250000")
    service: Optional[Literal["express", "standard"]]
    missing: list[str] = Field(description="Names of fields the customer did not give")

    @field_validator("cod_vnd")
    @classmethod
    def cod_in_vnd(cls, v):
        if v is not None and v % 1000:
            raise ValueError(f"must be in VND and a multiple of 1000, got {v}; 250k means 250000")
        if v is not None and v > COD_LIMIT:
            raise ValueError(f"{v} is above the COD limit of {COD_LIMIT} VND")
        return v

    @model_validator(mode="after")
    def status_matches_fields(self):
        empty = [f for f in FIELDS if getattr(self, f) is None]
        if self.status == "ok" and empty:
            raise ValueError(f"status is ok but {empty} are null; use need_info and list them in missing")
        if self.status == "ok" and self.pickup_district == self.dropoff_district:
            raise ValueError("pickup_district and dropoff_district must be different")
        return self


def error_text(err: ValidationError):
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'request'}: {e['msg']}" for e in err.errors())


def extract_with_retry(call, text, max_attempts=3, schema=Extraction):
    messages = [{"role": "user", "content": text}]
    errors = []
    for attempt in range(1, max_attempts + 1):
        reply = call(messages)
        try:
            return schema.model_validate_json(reply), attempt, errors
        except ValidationError as e:
            errors.append(error_text(e))
            # gửi lại chính câu trả lời sai và lỗi cụ thể, không chỉ "thử lại"
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"Your JSON is invalid: {errors[-1]}. Fix it and answer again."}]
    return None, max_attempts, errors
