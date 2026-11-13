                                                        

// Tách bộ đệm thành các sự kiện hoàn chỉnh; phần chưa đủ một sự kiện được trả lại trong rest.
export function parseSse(buffer        )                                       {
  // TODO 12: tách buffer theo dòng trống; mỗi khối có dòng "event: ..." và một hoặc nhiều dòng "data: ...";
  // bỏ qua dòng bắt đầu bằng ":"; data là JSON. Khối cuối chưa kết thúc bằng dòng trống trả lại trong rest.
  return { events: [], rest: buffer };
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
