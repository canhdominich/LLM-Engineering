import json
import subprocess
import sys
from pathlib import Path

from common.llm import FakeLLM
from fastapi.testclient import TestClient

import chat_api as ca

store = ca.PgStore().init()
llm = FakeLLM(["Chào Lan, mình đã ghi nhận đơn AT047838 giao tới Ben Song.", "Đơn AT047838 giao tới Ben Song."])
client = TestClient(ca.create_app(ca.FakeLLMProvider(llm), store))
r = client.post("/chat", json={"message": "Tôi là Lan, đơn AT047838 giao tới Ben Song."})
lines = [x for x in r.text.split("\n") if x]
cid = json.loads(lines[1][6:])["conversation_id"]
print(r.status_code, r.headers["content-type"])
print("\n".join(lines[:4] + ["..."] + lines[-2:]).replace(cid, "<conversation_id>"))
client.post("/chat", json={"conversation_id": cid, "message": "Đơn giao tới quận nào?"})
for m in client.get(f"/conversations/{cid}/messages").json():
    print(m["seq"], m["role"], m["content"], m["output_tokens"])
print("model saw on turn 2:", [m["role"] for m in llm.calls[1]["messages"]])
store.delete(cid)
tests = Path(__file__).resolve().parents[1] / "tests"
p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tests)], capture_output=True, text=True)
print(p.stdout.strip().splitlines()[-1])
