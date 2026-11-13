from dataclasses import dataclass

# Giá giả định để minh họa, đọc từ cấu hình chứ không viết cứng trong code nghiệp vụ
PRICES = {"small-model": {"input": 0.40, "output": 1.60}, "large-model": {"input": 2.50, "output": 10.00}}


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


def cost_usd(usage, model):
    p = PRICES[model]  # USD cho 1 triệu token
    return (usage.input_tokens * p["input"] + usage.output_tokens * p["output"]) / 1_000_000


calls = {
    "short answer (recorded)": Usage(68, 24),
    "thinking on (recorded)": Usage(62, 154),
    "turn 10 of a chat": Usage(399, 24),
}
for name, u in calls.items():
    small, large = cost_usd(u, "small-model"), cost_usd(u, "large-model")
    print(f"{name:<24} in={u.input_tokens:>5} out={u.output_tokens:>4}  small ${small:.6f}  large ${large:.6f}")

per_day = 10_000
for name in ("short answer (recorded)", "thinking on (recorded)"):
    month = cost_usd(calls[name], "large-model") * per_day * 30
    print(f"30 days x {per_day} calls, large-model, {name}: ${month:,.2f}")
u = calls["thinking on (recorded)"]
share = u.output_tokens * PRICES["large-model"]["output"] / (u.input_tokens * PRICES["large-model"]["input"] + u.output_tokens * PRICES["large-model"]["output"])
print(f"output share of the bill when thinking is on: {share:.1%}")
