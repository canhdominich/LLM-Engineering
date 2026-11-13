import io

from openai import OpenAI

from _common import BASE_URL

client = OpenAI(base_url=BASE_URL, api_key="ollama", max_retries=0)
line = b'{"custom_id": "M0001", "method": "POST", "url": "/v1/responses", "body": {"model": "qwen3:1.7b", "input": "hi"}}\n'
client.files.create(file=("intent_batch.jsonl", io.BytesIO(line)), purpose="batch")  # bước 1 của Batch API
