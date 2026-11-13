import random
from collections import Counter

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def should_retry(status):
    return status in RETRYABLE  # 400, 401, 403, 404, 422: gửi lại y hệt vẫn hỏng


def delay(attempt, rng, base=0.5, cap=8.0, jitter=True):
    ceiling = min(cap, base * 2 ** attempt)  # 0,5; 1; 2; 4; 8 giây
    return rng.uniform(0, ceiling) if jitter else ceiling


for status in (400, 401, 404, 408, 422, 429, 500, 503):
    print(f"status {status}: {'retry' if should_retry(status) else 'fail fast'}")

rng = random.Random(7)
for jitter in (False, True):
    buckets = Counter()
    for client in range(1000):  # 1.000 phiên chat cùng nhận 429 lúc t = 0
        t = 0.0
        for attempt in range(3):
            t += delay(attempt, rng, jitter=jitter)
            buckets[round(t, 1)] += 1
    peak_t, peak = buckets.most_common(1)[0]
    print(f"jitter={jitter!s:<5} busiest 100 ms window at {peak_t:.1f} s gets {peak} requests")

deadline, spent, attempt = 10.0, 0.0, 0
rng = random.Random(1)
while True:
    spent += 2.0  # mỗi lần thử bị timeout sau 2 giây
    d = delay(attempt, rng)
    if spent + d + 2.0 > deadline:
        print(f"deadline 10 s: stop after {attempt + 1} attempts, {spent:.1f} s spent, show a friendly error")
        break
    spent += d
    attempt += 1
