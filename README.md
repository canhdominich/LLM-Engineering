# LLM-Engineering

Mã nguồn lab của khóa "LLM Engineering: Hiểu Transformer và xây ứng dụng LLM" (chương trình AI Engineering của GDev). Thư mục `c01` đến `c15` là các chương, `project-chat` là dự án cuối khóa (ứng dụng chat hỗ trợ khách hàng).

## Chạy thử

Cần `uv`, Python 3.12, Docker (PostgreSQL với pgvector, Ollama với model `qwen3:1.7b` cho các ví dụ gọi LLM thật).

```bash
uv sync
docker run -d --name anhti-pg -p 55432:5432 -e POSTGRES_USER=anhti -e POSTGRES_PASSWORD=anhti -e POSTGRES_DB=anhti pgvector/pgvector:pg17
docker run -d --name anhti-ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama && docker exec anhti-ollama ollama pull qwen3:1.7b
```

Một số chương đọc `orders.parquet` của khóa AI Foundations: sinh bằng `make_dataset.py` trong repo AI-Foundations rồi đặt biến môi trường `ANHTI_DATA` trỏ tới thư mục chứa nó.

Mỗi chương có `examples/` và `lab/{starter,solution,tests}`. Test dùng `FakeLLM` nên không cần model. Dữ liệu là dữ liệu tổng hợp; xem `m02/data/DATA_CARD.md` về nhãn của bộ tin nhắn.
