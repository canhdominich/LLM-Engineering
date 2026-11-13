import json

# Model bị cắt vì max_output_tokens quá nhỏ: chuỗi arguments thiếu dấu đóng
raw_arguments = '{"order_id": "AT0478'
args = json.loads(raw_arguments)
print(args)
