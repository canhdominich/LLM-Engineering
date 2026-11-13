type ChatEvent = { event: string; data: any };

export async function* readSse(res: Response): AsyncGenerator<ChatEvent> {
  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) { // một sự kiện kết thúc bằng dòng trống
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      yield { event, data: JSON.parse(data.join("\n")) };
    }
  }
}

const url = process.argv[2];
const t0 = performance.now();
const res = await fetch(url, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ message: "Đơn AT047838 đâu rồi?" }),
});
let shown = "";
for await (const ev of readSse(res)) {
  if (ev.event === "delta") {
    if (!shown) console.log(`first text painted at ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    shown += ev.data.text; // trình duyệt thật sẽ gán vào DOM ở đây
  } else if (ev.event === "done") {
    console.log(`done at ${((performance.now() - t0) / 1000).toFixed(1)} s after ${ev.data.chunks} chunks`);
  }
}
console.log("screen:", shown);
