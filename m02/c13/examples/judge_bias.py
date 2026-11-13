from _common import human_labels, load_golden, replies
from judge import judge_checklist, judge_holistic, judge_pair

items = load_golden()
human = human_labels()
r1, r2 = replies("v1", items), replies("v2", items)

pairs = [it for it in items if human[(it["id"], "v1")] != human[(it["id"], "v2")]][:8]
consistent = correct = 0
for it in pairs:
    a, b = r1[it["id"]]["reply"], r2[it["id"]]["reply"]
    first = "v1" if judge_pair(it, a, b) == "A" else "v2"  # v1 đứng ở vị trí A
    second = "v2" if judge_pair(it, b, a) == "A" else "v1"  # đổi chỗ: v2 đứng ở vị trí A
    better = "v1" if human[(it["id"], "v1")] else "v2"
    consistent += first == second
    correct += first == second == better
    print(f"{it['id']} human prefers {better}: v1 in A picks {first}, v2 in A picks {second}")
print(f"position swap: same winner {consistent}/{len(pairs)}, same and correct {correct}/{len(pairs)}")

PAD = " Cảm ơn bạn rất nhiều vì đã tin tưởng Giao Hàng Anh Tí, chúng tôi luôn đặt sự hài lòng của bạn lên hàng đầu."
bad = [it for it in items if not human[(it["id"], "v1")]][:6]
flips = {"holistic": 0, "checklist": 0}
for it in bad:
    text = r1[it["id"]]["reply"]
    h0, h1 = judge_holistic(it, text), judge_holistic(it, text + PAD)
    c0, c1 = judge_checklist(it, text), judge_checklist(it, text + PAD)
    flips["holistic"] += h1["pass"] and not h0["pass"]
    flips["checklist"] += c1["pass"] and not c0["pass"]
    print(f"{it['id']} score {h0['detail']['score']} then {h1['detail']['score']} | checklist {c0['pass']} then {c1['pass']}")
print(f"padding one polite sentence onto {len(bad)} failing replies: holistic flips to pass {flips['holistic']}, "
      f"checklist flips to pass {flips['checklist']}")
