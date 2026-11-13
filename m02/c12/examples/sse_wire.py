import json


def frame_naive(text, event="delta"):
    return f"event: {event}\ndata: {text}\n\n"


def frame_json(obj, event="delta"):
    return f"event: {event}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"


def parse(raw):
    out = []
    for block in raw.split("\n\n"):  # dòng trống kết thúc một sự kiện
        if not block.strip():
            continue
        ev = {"event": "message", "data": []}
        for line in block.split("\n"):
            field, _, value = line.partition(": ")
            if field == "event":
                ev["event"] = value
            elif field == "data":
                ev["data"].append(value)
        out.append((ev["event"], "\n".join(ev["data"])))
    return out


deltas = ["Đơn AT047838", " đang ở", " Cho Moi.\n\nShipper", " sẽ gọi bạn."]
naive = "".join(frame_naive(d) for d in deltas)
print("naive wire:", repr(naive[:58]))
got = parse(naive)
print(f"naive: sent {len(deltas)} deltas, client parsed {len(got)} events")
print("   rebuilt:", repr("".join(d for _, d in got)))
safe = "".join(frame_json({"text": d}) for d in deltas) + frame_json({"output_tokens": 14}, "done")
got = parse(safe)
print(f"json : sent {len(deltas)} deltas + done, client parsed {len(got)} events")
print("   rebuilt:", repr("".join(json.loads(d)["text"] for e, d in got if e == "delta")))
print("   last event:", got[-1])
