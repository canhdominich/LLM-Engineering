import re
import unicodedata
from collections import Counter

import tiktoken

MARK = "▁"

PAIRS = [
    ("Đơn AT027075 của tôi đang ở đâu vậy?", "Where is my order AT027075 right now?"),
    ("Cho mình hỏi tình trạng đơn AT010570 với ạ.", "Could you tell me the status of order AT010570?"),
    ("App báo đơn đã giao nhưng tôi chưa nhận được gì cả.", "The app says the order was delivered but I have not received anything."),
    ("Shipper đã lấy hàng ở Tạp Hóa Cô Tấm chưa?", "Has the shipper picked up the parcel at Tap Hoa Co Tam yet?"),
    ("Khi nào đơn AT027075 tới nơi?", "When will order AT027075 arrive?"),
    ("Mấy giờ shipper tới Khu Công Nghiệp vậy?", "What time will the shipper reach Khu Cong Nghiep?"),
    ("Tôi đi vắng lúc 17 giờ, đơn có tới trước giờ đó không?", "I will be out at 5 pm. Will the order arrive before then?"),
    ("Giao từ Cà Phê Chú Tư tới Xóm Vui mất bao lâu?", "How long does delivery from Cafe Chu Tu to Xom Vui take?"),
    ("Đơn hỏa tốc mà trễ 40 phút, tôi rất thất vọng.", "An express order that is 40 minutes late. I am very disappointed."),
    ("Đồ ăn nguội hết rồi vì giao quá giờ hẹn.", "The food went cold because the delivery was past the promised time."),
    ("Khu Bến Sông lần nào cũng bị giao chậm.", "Deliveries to Ben Song are slow every single time."),
    ("Tôi muốn hoàn tiền đơn AT010570 vì hàng bị vỡ.", "I want a refund for order AT010570 because the item arrived broken."),
    ("Phí giao hàng bị trừ hai lần, xin hoàn lại một lần.", "The delivery fee was charged twice. Please refund one of them."),
    ("Cho mình đổi địa chỉ giao đơn AT027075 sang Chợ Mới.", "Please change the delivery address of order AT027075 to Cho Moi."),
    ("Tôi chuyển nhà sang Cầu Gỗ, cập nhật địa chỉ giúp tôi.", "I moved to Cau Go. Please update my address."),
    ("Phí giao hàng đi Bãi Biển là bao nhiêu?", "How much is the delivery fee to Bai Bien?"),
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def word_counts(texts: list[str]) -> Counter:
    return Counter(tuple(MARK + w) for t in texts for w in normalize(t).split())


def pair_counts(words: Counter) -> Counter:
    pairs = Counter()
    for symbols, n in words.items():
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += n
    return pairs


def merge(symbols: tuple, pair: tuple) -> tuple:
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(texts: list[str], num_merges: int) -> list[tuple[str, str]]:
    words = word_counts(texts)
    merges = []
    for _ in range(num_merges):
        pairs = pair_counts(words)
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        words = Counter({merge(s, best): n for s, n in words.items()})
    return merges


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    ranks = {pair: i for i, pair in enumerate(merges)}
    tokens = []
    for w in normalize(text).split():
        symbols = tuple(MARK + w)
        while len(symbols) > 1:
            best = min(zip(symbols, symbols[1:]), key=lambda p: ranks.get(p, float("inf")))
            if best not in ranks:
                break
            symbols = merge(symbols, best)
        tokens.extend(symbols)
    return tokens


def decode(tokens: list[str]) -> str:
    return "".join(tokens).replace(MARK, " ").strip()


def count_tokens(texts: list[str], encoding_name: str = "o200k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return sum(len(enc.encode(t)) for t in texts)


def language_ratio(pairs: list[tuple[str, str]], encoding_name: str = "o200k_base") -> float:
    vi = count_tokens([p[0] for p in pairs], encoding_name)
    en = count_tokens([p[1] for p in pairs], encoding_name)
    return vi / en


def conversation_tokens(system: str, turns: list[tuple[str, str]], count, overhead: int = 4) -> tuple[int, int]:
    history = count(system) + overhead
    tokens_in = tokens_out = 0
    for user, reply in turns:
        history += count(user) + overhead
        tokens_in += history
        tokens_out += count(reply) + overhead
        history += count(reply) + overhead
    return tokens_in, tokens_out


def estimate_cost(tokens_in: int, tokens_out: int, n_conversations: int, price_in: float, price_out: float) -> float:
    return n_conversations * (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def main() -> None:
    vi = [p[0] for p in PAIRS]
    merges = train_bpe(vi, 80)
    print(f"merges learned: {len(merges)}  first 5: {[a + b for a, b in merges[:5]]}")
    sample = "Đơn AT027075 trễ hẹn rồi"
    tokens = encode(sample, merges)
    print(f"own BPE: {len(tokens)} tokens {tokens}  roundtrip: {decode(tokens) == normalize(sample)}")
    own = sum(len(encode(t, merges)) for t in vi)
    print(f"own BPE tokens for {len(vi)} vi messages: {own}  characters: {sum(len(normalize(t)) for t in vi)}")
    for name in ("o200k_base", "cl100k_base"):
        print(f"{name}: vi/en token ratio = {language_ratio(PAIRS, name):.2f}")

    enc = tiktoken.get_encoding("o200k_base")
    count = lambda text: len(enc.encode(text))
    system = "Bạn là trợ lý chăm sóc khách hàng của Giao Hàng Anh Tí. Trả lời ngắn gọn, lịch sự."
    turns = [(PAIRS[0][0], "Đơn AT027075 đang được giao, dự kiến tới trong 25 phút nữa."),
             (PAIRS[6][0], "Có ạ, đơn dự kiến tới lúc 16 giờ 35."),
             (PAIRS[13][0], "Đơn đã rời kho nên không đổi được địa chỉ.")]
    t_in, t_out = conversation_tokens(system, turns, count)
    cost = estimate_cost(t_in, t_out, 10_000, price_in=0.40, price_out=1.60)
    print(f"one conversation: {t_in} input + {t_out} output tokens; 10,000 conversations: {cost:.2f} USD (illustrative prices)")


if __name__ == "__main__":
    main()
