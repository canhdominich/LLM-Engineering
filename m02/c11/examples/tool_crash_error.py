import json

from _shop import FUNCTIONS

# Model đọc nhầm mã đơn của khách và xin tra một đơn không tồn tại
call = {"name": "get_order_status", "arguments": '{"order_id": "AT999999"}'}
result = FUNCTIONS[call["name"]](**json.loads(call["arguments"]))
print(result)
