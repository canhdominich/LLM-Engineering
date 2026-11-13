import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bot import ACTIVE, MODEL, answer
from checks import summarize
from golden import load_golden

METRICS = ["intent_acc", "macro_f1", "fact_rate", "policy_rate"]


def gate(report, baseline, cfg, items):
    reasons = []
    for k, lo in cfg["min"].items():
        if report[k] < lo:
            reasons.append(f"{k} {report[k]:.3f} below min {lo}")
    was_ok = {r["id"]: r["rules_ok"] for r in baseline["rows"]}
    broken = [r["id"] for r in report["rows"] if was_ok.get(r["id"]) and not r["rules_ok"]]
    if len(broken) > cfg["max_broken"]:
        reasons.append(f"{len(broken)} items broken vs baseline (max {cfg['max_broken']}): {broken}")
    hard = {it["id"] for it in items if set(it["tags"]) & set(cfg["hard_tags"])}
    known = set(cfg.get("known_failures", []))  # lỗi đã biết, có người nhận sửa, ghi rõ trong cấu hình
    leaks = [r["id"] for r in report["rows"] if r["id"] in hard - known and not r["policy_ok"]]
    if leaks:
        reasons.append(f"policy failed on hard items: {leaks}")
    return reasons


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=ACTIVE)
    ap.add_argument("--baseline", default=str(HERE / "baseline.json"))
    ap.add_argument("--config", default=str(HERE / "eval_config.json"))
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args(argv)
    items = load_golden()
    preds = {it["id"]: answer(it, args.prompt) for it in items}
    report = summarize(items, preds)
    if args.write_baseline:
        Path(args.baseline).write_text(json.dumps({"prompt": args.prompt, "model": MODEL, **report}, indent=1))
        print("baseline written:", args.baseline)
        return 0
    baseline = json.loads(Path(args.baseline).read_text())
    cfg = json.loads(Path(args.config).read_text())
    lines = [f"| metric | baseline {baseline['prompt']} | {args.prompt} |", "| --- | --- | --- |"]
    lines += [f"| {k} | {baseline[k]:.3f} | {report[k]:.3f} |" for k in METRICS]
    reasons = gate(report, baseline, cfg, items)
    lines.append(f"\nGate: {'PASS' if not reasons else 'FAIL'} ({report['n']} items, model {MODEL})")
    lines += [f"- {r}" for r in reasons]
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):  # GitHub Actions hiện bảng này trên trang của lần chạy
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
