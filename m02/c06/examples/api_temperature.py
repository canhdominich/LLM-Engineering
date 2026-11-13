import os

from openai import OpenAI

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"), api_key="ollama", timeout=300)
MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
PROMPT = "A customer of Giao Hang Anh Ti asks: 'My order AT027075 is late, where is it?' Reply in one short sentence."

for temperature in (0.0, 0.0, 1.2, 1.2):
    r = client.responses.create(model=MODEL, input=PROMPT, temperature=temperature, top_p=1.0,
                                max_output_tokens=60, reasoning={"effort": "none"})
    print(f"T={temperature}: {r.output_text.strip()}")
print("model:", MODEL)
