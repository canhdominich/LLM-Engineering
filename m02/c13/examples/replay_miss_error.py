import os
import sys
from pathlib import Path

os.environ["C13_REPLAY"] = "1"  # chế độ phát lại: chỉ đọc câu trả lời đã ghi, không gọi model
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import bot
from run_eval import main

bot.PROMPTS["v3"] = bot.V1.replace("at most three short sentences", "at most two short sentences")
main(["--prompt", "v3"])
