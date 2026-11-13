# anhti_chat: chat hỗ trợ khách hàng cho Giao Hàng Anh Tí

Bản tham khảo của dự án cuối khóa LLM Engineering. Dữ liệu là mô phỏng: `data/orders.csv` (sinh bằng `data/make_orders.py`, seed 15), dự đoán giờ giao đóng băng từ mô hình P1 của mô-đun 01 (`data/make_eta_snapshot.py`), 40 hội thoại đánh giá (`evals/make_conversations.py`).

## Chạy

```bash
cd m02/project-chat
PYTHONPATH=solution:../.. uv run pytest -q tests                  # 42 test, cần PostgreSQL ở ANHTI_PG_DSN
cd solution/frontend && node build.mjs && cd ../..                 # build static/*.js từ src/*.ts (Node 22)
cd solution && PYTHONPATH=.:../../.. uv run python -m anhti_chat serve   # http://127.0.0.1:8000
```

Biến môi trường:

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai` hoặc `fake` |
| `LLM_MODEL` | `qwen3:1.7b` | tên model của provider |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | endpoint tương thích OpenAI |
| `OPENAI_API_KEY` | | chỉ ở server, không bao giờ ở frontend |
| `ANHTI_PG_DSN` | `postgresql://anhti:anhti@localhost:55432/anhti` | PostgreSQL; ứng dụng dùng schema `m02_project`, test dùng `m02_project_test` |
| `ANHTI_ETA_URL` | | URL dịch vụ P1 (`POST /predict`); bỏ trống thì dùng snapshot |
| `ANHTI_HISTORY_BUDGET` | `300` | ngân sách token cho lịch sử |
| `ANHTI_MAX_TOOL_STEPS` | `5` | giới hạn vòng lặp công cụ |

## Đánh giá

```bash
PYTHONPATH=solution:../.. uv run python -m anhti_chat eval            # phát lại evals/cache, in metric và kết quả cổng
PYTHONPATH=solution:../.. uv run python -m anhti_chat eval --record   # gọi model thật, ghi lại cache (lâu)
```

Ngưỡng nằm trong `evals/thresholds.toml`. Sửa prompt, công cụ hoặc schema làm khóa cache đổi: phải ghi lại bằng `--record` trước khi hợp nhất.

## Giới hạn đã biết

- Đăng nhập là token demo cố định trong `config.py`.
- Lớp kiểm tra câu trả lời so cụm từ sau khi bỏ dấu, nên có báo nhầm (câu "không thể hoàn tiền" cũng bị thay).
- Bộ đánh giá dùng `MemoryStore` và snapshot P1 để tách phần LLM khỏi database và mô hình ETA.
- TTFT đo trên `qwen3:1.7b` chạy CPU dùng chung, không đại diện cho provider API.
