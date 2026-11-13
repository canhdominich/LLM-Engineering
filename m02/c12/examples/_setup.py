import os
import time

from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
INSTRUCTIONS = "Bạn là trợ lý chat của Giao Hang Anh Ti ở Pho Rom Ra. Trả lời ngắn gọn bằng tiếng Việt, tối đa hai câu."
DSN = os.environ.get("ANHTI_PG_DSN", "postgresql://anhti:anhti@localhost:55432/anhti")
SCHEMA = "m02_c12"


def client(**kw):
    kw.setdefault("timeout", 180)
    return OpenAI(base_url=BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "ollama"), **kw)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
