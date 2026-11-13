import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
EVALS_DIR = PROJECT_DIR / "evals"

DEMO_USERS = {
    "token-an": ("C01024", "An"),
    "token-binh": ("C02210", "Binh"),
    "token-lan": ("C03357", "Lan"),
    "token-minh": ("C04481", "Minh"),
    "token-ha": ("C05112", "Ha"),
}


@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"
    model: str = "qwen3:1.7b"
    base_url: str | None = "http://localhost:11434/v1"
    api_key: str = "ollama"
    pg_dsn: str = "postgresql://anhti:anhti@localhost:55432/anhti"
    pg_schema: str = "m02_project"
    orders_path: Path = DATA_DIR / "orders.csv"
    eta_url: str | None = None
    eta_snapshot: Path = DATA_DIR / "eta_snapshot.json"
    now: str = "2026-11-05T10:30:00"
    max_tool_steps: int = 5
    history_budget_tokens: int = 300
    max_output_tokens: int = 300
    llm_timeout_s: float = 120.0

    @property
    def clock(self) -> datetime:
        return datetime.fromisoformat(self.now)


ENV = {
    "LLM_PROVIDER": "provider", "LLM_MODEL": "model", "LLM_BASE_URL": "base_url", "OPENAI_API_KEY": "api_key",
    "ANHTI_PG_DSN": "pg_dsn", "ANHTI_PG_SCHEMA": "pg_schema", "ANHTI_ORDERS": "orders_path", "ANHTI_ETA_URL": "eta_url",
    "ANHTI_NOW": "now", "ANHTI_MAX_TOOL_STEPS": "max_tool_steps", "ANHTI_HISTORY_BUDGET": "history_budget_tokens",
}


def load_settings(env: dict | None = None, **overrides) -> Settings:
    env = os.environ if env is None else env
    values: dict = {}
    for name, field in ENV.items():
        if env.get(name):
            values[field] = env[name]
    if values.get("provider") == "openai" and "base_url" not in values:
        values["base_url"] = None
    values.update(overrides)
    for key in ("max_tool_steps", "history_budget_tokens"):
        if key in values:
            values[key] = int(values[key])
    if "orders_path" in values:
        values["orders_path"] = Path(values["orders_path"])
    return replace(Settings(), **values)
