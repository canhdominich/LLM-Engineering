import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

POOL = ThreadPoolExecutor(max_workers=4)


def slow_predict_eta(order_id):
    time.sleep(2.0)  # mô phỏng: dịch vụ dự đoán bị nghẽn
    return {"order_id": order_id, "minutes_left": 11}


def fast_status(order_id):
    return {"order_id": order_id, "status": "out_for_delivery"}


def run_with_timeout(fn, arg, timeout_s):
    t0 = time.perf_counter()
    future = POOL.submit(fn, arg)
    try:
        result = future.result(timeout=timeout_s)
    except TimeoutError:
        future.cancel()  # không dừng được luồng đang chạy, chỉ bỏ kết quả
        result = {"error": "timeout", "message": f"{fn.__name__} took longer than {timeout_s} s", "retryable": True}
    return result, time.perf_counter() - t0


for fn in (fast_status, slow_predict_eta):
    result, waited = run_with_timeout(fn, "AT047838", timeout_s=0.5)
    print(f"{fn.__name__:<17} waited {waited:.1f} s:", result)
POOL.shutdown(wait=False, cancel_futures=True)
