import _project  # noqa: F401

import anhti_chat.chat as chat
from anhti_chat.evals import check_gate, load_thresholds, run_eval

th = load_thresholds()
base = run_eval()
print("baseline        gate:", check_gate(base["metrics"], th) or "pass")

no_guard = run_eval(guard=False)
m = no_guard["metrics"]
print(f"guard removed   violations={m['safety_violations']} ungrounded_minutes={m['ungrounded_minutes']} fact_accuracy={m['fact_accuracy']}")
print("                gate:", check_gate(m, th))
for r in no_guard["rows"]:
    if r["violations"] or r["ungrounded"]:
        print(f"    {r['id']} {r['violations'] + [f'ungrounded:{n}' for n in r['ungrounded']]} answer={r['answer'][:70]!r}")

chat.SYSTEM += "\nAlways greet the customer by name."
edited = run_eval()
errors = sum(r["error"] == "replay_miss" for r in edited["rows"])
print(f"prompt edited   replay misses in {errors} of 40 conversations, gate:", check_gate(edited["metrics"], th)[:2])
