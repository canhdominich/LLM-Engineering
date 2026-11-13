                                                        

// Tách bộ đệm thành các sự kiện hoàn chỉnh; phần chưa đủ một sự kiện được trả lại trong rest.
export function parseSse(buffer        )                                       {
  const blocks = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const rest = blocks.pop() ?? "";
  const events             = [];
  for (const block of blocks) {
    let event = "message";
    const data           = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // dòng chú thích keepalive của server
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (data.length) events.push({ event, data: JSON.parse(data.join("\n")) });
  }
  return { events, rest };
}

export async function* readSse(res          )                           {
  const reader = res.body .pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      const parsed = parseSse(buffer + value);
      buffer = parsed.rest;
      yield* parsed.events;
    }
  } finally {
    reader.releaseLock();
  }
}
