def estimate_tokens(text: str) -> int:
    # ước lượng thô cho tiếng Việt với tokenizer của Qwen3: khoảng 3 ký tự một token, cộng phần khung của một tin nhắn
    return len(text) // 3 + 4


def build_context(messages: list[dict], summary_upto: int, budget: int, count=estimate_tokens) -> tuple[list[dict], list[dict]]:
    """Return (recent items to send, older messages that should be folded into the summary).

    Only whole turns are kept, newest first, and the last user message is always kept."""
    # TODO 5: bỏ các tin đã nằm trong tóm tắt (seq <= summary_upto) và tin rỗng; giữ các lượt trọn vẹn mới nhất
    # (một lượt bắt đầu bằng tin user) sao cho tổng count(content) không vượt budget; lượt cuối luôn được giữ.
    # Trả về (items dạng {"role", "content"}, danh sách tin cũ hơn cần tóm tắt).
    raise NotImplementedError("TODO 5")
