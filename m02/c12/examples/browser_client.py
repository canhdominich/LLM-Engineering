import subprocess

from _sse_app import app
from _fake_stream import serve

server, base = serve(app)
r = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", "chat_client.ts", base + "/chat"],
                   capture_output=True, text=True, timeout=30)
print(r.stdout.strip() or r.stderr[-800:])
server.should_exit = True
