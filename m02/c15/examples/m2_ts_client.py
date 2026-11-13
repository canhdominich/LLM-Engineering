import subprocess
import time

import _project  # noqa: F401
from _server import last_assistant, serve, slow_service

from anhti_chat.api import create_app

service, llm = slow_service(rounds=2)
server, base = serve(create_app(service))
for stop_after in (0, 6):
    p = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", "ts_client.ts", base, str(stop_after)],
                       capture_output=True, text=True, timeout=60)
    print(f"--- stop after {stop_after or 'never'}")
    print(p.stdout.strip() or p.stderr[-600:])
    time.sleep(1.0)
    row = last_assistant(service.store)
    print(f"server saved: status={row['status']}, {len(row['content'].split())} words")
server.should_exit = True
