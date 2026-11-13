import { readSse } from "../../project-chat/solution/frontend/src/sse.ts";

const [base, stopAfter] = [process.argv[2], Number(process.argv[3])];
const controller = new AbortController();
const t0 = performance.now();
const at = () => ((performance.now() - t0) / 1000).toFixed(1).padStart(4);
let words = 0;
let screen = "";
try {
  const res = await fetch(base + "/chat", {
    method: "POST",
    signal: controller.signal,
    headers: { "content-type": "application/json", authorization: "Bearer token-an" },
    body: JSON.stringify({ message: "Đơn AT047838 đâu rồi?" }),
  });
  for await (const ev of readSse(res)) {
    const d = ev.data as Record<string, any>;
    if (ev.event === "delta") {
      screen += d.text;
      if (++words === stopAfter) {
        console.log(`${at()} s  user clicks Stop after ${words} words`);
        controller.abort();
      }
    } else {
      const shown = ev.event === "meta" ? { ...d, conversation_id: "<id>" } : d;
      console.log(`${at()} s  ${ev.event} ${JSON.stringify(shown).slice(0, 70)}`);
    }
  }
} catch (e) {
  console.log(`${at()} s  ${(e as Error).name}: fetch aborted`);
}
console.log(`screen: ${screen.split(" ").length} words, ${JSON.stringify(screen.slice(0, 40))}`);
