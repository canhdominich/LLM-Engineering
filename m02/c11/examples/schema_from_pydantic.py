import json

from pydantic import BaseModel, ConfigDict, Field


class OrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")  # sinh "additionalProperties": false
    order_id: str = Field(pattern=r"^AT\d{6}$", description="Order id: AT followed by 6 digits, for example AT047838")


schema = OrderArgs.model_json_schema()
schema.pop("title")
schema["properties"]["order_id"].pop("title")

responses_tool = {"type": "function", "name": "get_order_status", "strict": True,
                  "description": "Return where an order is now. Use for 'where is my order'. It does not give the arrival time.",
                  "parameters": schema}
# Chat Completions bọc cùng nội dung trong một khóa "function"
chat_tool = {"type": "function", "function": {k: v for k, v in responses_tool.items() if k != "type"}}

print(json.dumps(responses_tool, indent=1))
print("chat completions keys:", list(chat_tool), "inner:", list(chat_tool["function"]))
print("description length:", len(responses_tool["description"]), "characters")
