# Data card: golden set of the Giao Hang Anh Ti support chat (m02/c13)

`golden.jsonl` is written by `make_golden.py` (no randomness: the 40 items are listed in the script). 30 items reuse a customer message of `lab/m02/data/messages.csv` (test split, read only, field `source` keeps the message id); 10 items are hand written (`source: handwritten`). Every order is fictional and does not join to `orders.parquet`.

Fields of one line:
- `id`: G01 to G40.
- `tags`: `single_turn` or `multi_turn`, `diacritics` or `no_diacritics`, plus `typo`, `missing_id` (an order question without order id, the bot must ask for it), `injection` (the customer tries to change the rules), `hard` (label rule or delivered-but-not-received).
- `order`: the facts the bot is allowed to use (as if a tool had looked them up) or `null`.
- `turns`: the conversation, last turn always from the customer.
- `expected.intent`: one of the six intents of `messages.csv`, same labelling rule.
- `expected.must`: list of groups; the reply must contain at least one string of every group, compared after removing diacritics and case.
- `expected.forbid`: strings that must not appear (promises of refund, compensation, voucher; claims that an address was changed).
- `reference`: one reply that passes every check, written by hand.

Counts: order_status 7, eta 8, complaint_late 6, refund 7, change_address 6, other 6. Tags: diacritics 25, no_diacritics 15, typo 6, missing_id 6, multi_turn 5, injection 3, hard 2.

Other files:
- `golden_sample.jsonl`: items G06, G25 and G37, copied from `golden.jsonl` for the lab instructions.
- `llm_cache/`: replies of qwen3:1.7b (Ollama 0.40, temperature 0, reasoning off) for the bot prompts v1 and v2 and for the judges, recorded in the night of 2026-11-05 to 2026-11-06. The key is a hash of model, prompt, input and settings, so a changed prompt never reuses an old reply.
- `replies_v1.jsonl`, `replies_v2.jsonl`: the bot replies exported from the cache, one per item.
- `human_labels.csv`: one pass or fail label per reply (80 rows), assigned during course authoring with the help of an AI authoring tool (not yet reviewed by a human), reading ORDER, POLICY, the conversation and the reply, with the three criteria of the LLM-as-a-judge lesson: facts correct and nothing invented, answers the last message (asks for the id when needed), follows the policy. The `note` column says why a reply fails. One labelling pass: there is no second labeller and no human review yet, so the labels themselves carry an unknown error.
- `judge_checklist.jsonl`: verdicts of the checklist judge for the 80 replies.

Known limits: 40 items is small, one item moves a rate by 2.5 points; the `must` and `forbid` checks are keyword rules and miss paraphrases ("không thể xác nhận hoàn tiền" first matched the forbidden "xác nhận hoàn"; fixed by checking the references, see `examples/golden_overview.py`); conversations are short and cleaner than real chats.
