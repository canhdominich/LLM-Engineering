import time
from concurrent.futures import ThreadPoolExecutor

from _shop import FUNCTIONS, eta_model

eta_model()  # huấn luyện mô hình trước để không tính vào thời gian đo
NETWORK_S = 0.4  # mô phỏng: mỗi công cụ gọi một dịch vụ nội bộ mất 0,4 giây


def run(call):
    time.sleep(NETWORK_S)
    return FUNCTIONS[call[0]](call[1])


calls = [("get_order_status", "AT047838"), ("predict_eta", "AT047838"),
         ("get_order_status", "AT051122"), ("predict_eta", "AT051122")]

t0 = time.perf_counter()
serial = [run(c) for c in calls]
t_serial = time.perf_counter() - t0

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as pool:
    parallel = list(pool.map(run, calls))  # map giữ đúng thứ tự call
t_parallel = time.perf_counter() - t0

print(f"serial  : {t_serial:.1f} s for {len(calls)} calls")
print(f"parallel: {t_parallel:.1f} s for {len(calls)} calls")
print("same results in same order:", serial == parallel)
for c, r in zip(calls, parallel):
    print(" ", c[0], c[1], {k: v for k, v in r.items() if k in ("status", "minutes_left")})
