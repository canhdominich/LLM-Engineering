import json
import time

from openai import OpenAI

from _common import BASE_URL, CACHE, few_shot_prompt, golden, wrap

client = OpenAI(base_url=BASE_URL, api_key="ollama", timeout=900, max_retries=0)
STABLE = few_shot_prompt(6)  # hướng dẫn + 36 ví dụ, giống hệt nhau ở mọi lời gọi
LOG = CACHE / "cache-layout.json"


def call(instructions, user_input):
    t0 = time.perf_counter()
    r = client.responses.create(model="qwen3:1.7b", instructions=instructions, input=user_input,
                                reasoning={"effort": "none"}, max_output_tokens=12, temperature=0)
    u = r.usage
    return {"in": u.input_tokens, "cached": u.input_tokens_details.cached_tokens or 0,
            "seconds": round(time.perf_counter() - t0, 2), "label": r.output_text.strip()}


def experiment():
    rows, out = golden().head(5), []
    for layout in ("stable first", "stamp first"):
        for i, r in enumerate(rows.itertuples()):
            stamp = f"Request {r.message_id} received at 2026-11-05 09:{i:02d}:00."
            if layout == "stable first":
                rec = call(STABLE, stamp + "\n" + wrap(r.text))
            else:
                rec = call(stamp + "\n" + STABLE, wrap(r.text))
            out.append(dict(rec, layout=layout, message_id=r.message_id))
    return out


log = json.loads(LOG.read_text()) if LOG.exists() else experiment()  # đo một lần, lưu lại số đo
LOG.write_text(json.dumps(log, indent=0))
print(f"{'layout':13}{'msg':>6}{'in tok':>8}{'cached':>8}{'seconds':>9}  label")
for x in log:
    print(f"{x['layout']:13}{x['message_id']:>6}{x['in']:>8}{x['cached']:>8}{x['seconds']:>9.2f}  {x['label']}")
for layout in ("stable first", "stamp first"):
    part = [x for x in log if x["layout"] == layout]
    share = sum(x["cached"] for x in part) / sum(x["in"] for x in part)
    print(f"{layout}: cached share {share:.1%}, total {sum(x['seconds'] for x in part):.1f} s for 5 calls")
