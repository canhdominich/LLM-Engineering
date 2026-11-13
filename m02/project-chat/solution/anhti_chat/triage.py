import json
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from .llm import collect

Intent = Literal["order_status", "eta", "complaint_late", "refund", "change_address", "other"]
HANDOFF_INTENTS = {"refund", "change_address"}


def _upper(v):
    return v.strip().upper() if isinstance(v, str) else v


OrderId = Annotated[str, Field(pattern=r"^AT\d{6}$")]


class Triage(BaseModel):
    model_config = ConfigDict(extra="forbid", title="Triage")
    intent: Intent
    order_id: Annotated[OrderId | None, BeforeValidator(_upper)]
    needs_human: bool
    reason: str = Field(max_length=160)


TRIAGE_SCHEMA = Triage.model_json_schema()

TRIAGE_PROMPT = """Classify the last customer message of a delivery support chat (Giao Hang Anh Ti).
intent:
- order_status: where the order is, whether it was picked up, delivered or cancelled
- eta: when it will arrive, how long it takes
- complaint_late: the order is late and the customer complains, without asking for money back
- refund: asks for money back or compensation, even if the order is late
- change_address: change address, receiver or drop-off point
- other: anything else
order_id: the order id mentioned in the conversation (AT + 6 digits), or null.
needs_human: true only if the customer explicitly asks for a human agent.
reason: at most 12 words, in English.
Reply with JSON only."""


def classify(client, instructions_extra: str, items: list[dict], max_attempts: int = 2):
    """Return (Triage, attempts, usages). Invalid output is retried once with the error, then falls back to a handoff."""
    usages, msgs = [], list(items)
    for attempt in range(1, max_attempts + 1):
        text, _, done = collect(client.stream(TRIAGE_PROMPT + instructions_extra, msgs, schema=TRIAGE_SCHEMA, max_output_tokens=150))
        if done:
            usages.append(done)
        try:
            return Triage.model_validate_json(text), attempt, usages
        except ValidationError as e:
            err = e.errors()[0]
            msgs = msgs + [{"role": "assistant", "content": text},
                           {"role": "user", "content": f"Invalid JSON for field {'.'.join(map(str, err['loc']))}: {err['msg']}. "
                                                       "Reply again with valid JSON only."}]
    return Triage(intent="other", order_id=None, needs_human=True, reason="triage_failed"), max_attempts, usages


def handoff_reason(t: Triage) -> str | None:
    if t.reason == "triage_failed":
        return "triage_failed"
    if t.intent in HANDOFF_INTENTS:
        return f"{t.intent} needs an agent"
    if t.needs_human:
        return "customer asked for a human"
    return None


def dump(t: Triage) -> dict:
    return json.loads(t.model_dump_json())
