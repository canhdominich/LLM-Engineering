import os
import subprocess
import sys

import _project  # noqa: F401
from _project import PROJECT

env = dict(os.environ, PYTHONPATH=f"{PROJECT / 'solution'}:{PROJECT.parents[1]}")
tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=PROJECT, env=env,
                       capture_output=True, text=True)
print("pytest:", tests.stdout.strip().splitlines()[-1].split(" in ")[0])
ev = subprocess.run([sys.executable, "-m", "anhti_chat", "eval"], cwd=PROJECT, env=env, capture_output=True, text=True)
print("eval exit code:", ev.returncode)
print(ev.stdout.strip())
