import { readSse } from "./sse.js";

                                

const $ =                        (id        )    => document.getElementById(id)     ;
const log = $                ("log");
const form = $                 ("form");
const input = $                  ("message");
const send = $                   ("send");
const stop = $                   ("stop");
const user = $                   ("user");
const status = $                      ("status");

let conversationId                = null;
let controller                         = null;

function bubble(role                      , text        )                       {
  const p = document.createElement("p");
  p.className = role;
  p.textContent = text; // textContent, không dùng innerHTML: câu trả lời của model là dữ liệu không đáng tin
  log.append(p);
  return p;
}

function seconds(t0        )         {
  return ((performance.now() - t0) / 1000).toFixed(1);
}

function setBusy(busy         )       {
  send.disabled = busy;
  stop.disabled = !busy;
  input.disabled = busy;
}

function showError(target                      , text        , retryable         , message        )       {
  const note = document.createElement("span");
  note.className = "error";
  note.textContent = " " + text;
  target.append(note);
  if (retryable) {
    const again = document.createElement("button");
    again.type = "button";
    again.textContent = "Thử lại";
    again.addEventListener("click", () => { again.remove(); void ask(message); });
    target.append(again);
  }
}

async function ask(message        )                {
  controller = new AbortController();
  setBusy(true);
  bubble("user", message);
  const answer = bubble("assistant", "");
  const t0 = performance.now();
  try {
    const res = await fetch("/chat", {
      method: "POST",
      signal: controller.signal,
      headers: { "content-type": "application/json", authorization: `Bearer ${user.value}` },
      body: JSON.stringify({ conversation_id: conversationId, message }),
    });
    if (!res.ok) {
      if (res.status === 404) conversationId = null;
      showError(answer, `Lỗi ${res.status}.`, false, message);
      return;
    }
    for await (const ev of readSse(res)) {
      const d = ev.data        ;
      if (ev.event === "meta") conversationId = d.conversation_id;
      else if (ev.event === "tool") status.textContent = `Đang tra cứu ${d.name}...`;
      else if (ev.event === "delta") {
        if (!answer.textContent) status.textContent = `Chữ đầu tiên sau ${seconds(t0)} s`;
        answer.textContent += d.text;
      } else if (ev.event === "replace") {
        answer.textContent = d.text;
        answer.classList.add("replaced");
      } else if (ev.event === "handoff") answer.dataset.handoff = `HO-${d.id}`;
      else if (ev.event === "error") {
        const text = d.retryable ? "Hệ thống đang bận, bạn thử lại sau ít phút." : "Có lỗi xảy ra, mình đã ghi nhận.";
        showError(answer, text, d.retryable, message);
      } else if (ev.event === "done") {
        status.textContent = `Xong sau ${seconds(t0)} s, ${d.calls} lời gọi model, ${d.input_tokens} + ${d.output_tokens} token`;
      }
    }
  } catch (e) {
    // TODO 13: nếu lỗi là AbortError (người dùng bấm Dừng) thì ghi "(đã dừng)", không hiện lỗi mất kết nối.
    showError(answer, "Mất kết nối tới máy chủ.", true, message);
  } finally {
    controller = null;
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message || controller) return;
  input.value = "";
  void ask(message);
});
// TODO 13: nút Dừng gọi abort() trên controller của yêu cầu đang chạy.
user.addEventListener("change", () => { conversationId = null; log.replaceChildren(); status.textContent = ""; });
