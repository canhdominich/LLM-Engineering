from _common import by_id, human_labels, load_golden, replies

items = load_golden()
golden, human = by_id(items), human_labels()
PICK = ["G01", "G06", "G07", "G10", "G30"]
TAGS = ["diacritics", "no_diacritics", "typo", "missing_id", "multi_turn", "injection"]
r2 = replies("v2", items)
for gid in PICK:
    print(f"{gid} [{'pass' if human[(gid, 'v2')] else 'fail'}] {golden[gid]['turns'][-1]['content'][:38]!r}: {r2[gid]['reply'][:58]}")
print(f"vibe check on v2: {sum(human[(g, 'v2')] for g in PICK)}/{len(PICK)} look fine")
for v in ["v1", "v2"]:
    total = sum(human[(it["id"], v)] for it in items)
    print(f"{v} on 40 golden items: {total}/40 pass")
print(f"{'slice':14}{'n':>3}{'v1':>5}{'v2':>5}")
for tag in TAGS:
    ids = [it["id"] for it in items if tag in it["tags"]]
    print(f"{tag:14}{len(ids):>3}{sum(human[(i, 'v1')] for i in ids):>5}{sum(human[(i, 'v2')] for i in ids):>5}")
broken = [it["id"] for it in items if human[(it["id"], "v1")] and not human[(it["id"], "v2")]]
fixed = [it["id"] for it in items if human[(it["id"], "v2")] and not human[(it["id"], "v1")]]
print("broken by v2:", broken)
print("fixed by v2:", fixed)
