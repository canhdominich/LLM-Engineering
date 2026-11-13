import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="anhti_chat")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    e = sub.add_parser("eval")
    e.add_argument("--record", action="store_true", help="call the real provider and save replies to evals/cache")
    e.add_argument("--only", default=None, help="comma separated conversation ids")
    e.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.cmd == "serve":
        import uvicorn
        from .api import build_service, create_app
        uvicorn.run(create_app(build_service()), host="127.0.0.1", port=args.port)
        return 0
    from .evals import check_gate, load_thresholds, run_eval
    report = run_eval(record=args.record, only=args.only.split(",") if args.only else None)
    failures = check_gate(report["metrics"], load_thresholds())
    summary = {"metrics": report["metrics"], "gate": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
