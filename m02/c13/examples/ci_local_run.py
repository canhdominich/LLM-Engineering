import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for prompt in ["v1", "v2"]:
    cmd = ["python", "evals/run_eval.py", "--prompt", prompt, "--baseline", "evals/baseline.json"]
    r = subprocess.run([sys.executable] + cmd[1:], cwd=ROOT, capture_output=True, text=True)
    print("$", " ".join(cmd))
    print(r.stdout.strip())
    print("exit code:", r.returncode, "\n")
