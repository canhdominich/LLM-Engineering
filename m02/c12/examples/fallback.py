from _providers import Delta, FakeProvider, ProviderError, ResponsesProvider


class Fallback:
    def __init__(self, primary, backup):
        self.primary, self.backup, self.name = primary, backup, f"{primary.name}|{backup.name}"

    def stream(self, messages, max_tokens=200):
        started = False
        try:
            for item in self.primary.stream(messages, max_tokens):
                started = started or isinstance(item, Delta)
                yield item
        except ProviderError as e:
            if started or not e.retryable:
                raise  # đã gửi chữ cho người dùng thì không âm thầm đổi model giữa câu
            print(f"   primary failed with {e.code}, switching to {self.backup.name}")
            yield from self.backup.stream(messages, max_tokens)


class BreaksMidway(FakeProvider):
    def stream(self, messages, max_tokens=200):
        for i, item in enumerate(super().stream(messages, max_tokens)):
            if i == 2:
                raise ProviderError("APIConnectionError", True, "connection reset")
            yield item


msgs = [{"role": "user", "content": "Đơn AT047838 đâu?"}]
down = ResponsesProvider("ollama-down", "qwen3:1.7b", "http://127.0.0.1:9/v1", "x", timeout=2.0)  # cổng không có ai nghe
backup = FakeProvider(["Hệ thống đang bận, đơn AT047838 vẫn đang được giao."])
text = "".join(i.text for i in Fallback(down, backup).stream(msgs) if isinstance(i, Delta))
print("case 1, primary down before any text:", text)
partial = ""
try:
    for item in Fallback(BreaksMidway(["Đơn AT047838 đang ở Cho Moi."]), FakeProvider(["unused"])).stream(msgs):
        partial += item.text if isinstance(item, Delta) else ""
except ProviderError as e:
    print(f"case 2, failed after {partial!r}: raised {e.code}, retryable={e.retryable}")
