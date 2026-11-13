from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

District = Literal["Dau Pho", "Cau Go", "Xom Vui", "Ben Song", "Bai Bien", "Cho Moi", "Doi Thong", "Khu Cong Nghiep"]


class DeliveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pickup_district: District = Field(description="District where the shipper picks up the goods")
    dropoff_district: District = Field(description="District where the goods are delivered")
    items: int = Field(ge=1, le=30, description="Number of packages")
    cod_vnd: int = Field(ge=0, description="Cash to collect from the receiver in VND, 0 if prepaid")
    service: Literal["express", "standard"] = Field(description="express only if the customer asks for urgent delivery")
