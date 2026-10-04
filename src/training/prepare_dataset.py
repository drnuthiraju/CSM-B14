from __future__ import annotations

import json
from pathlib import Path


def to_instruction(record: dict) -> str:
    return instruction_prompt(record) + label_from_record(record)


def label_from_record(record: dict) -> str:
    """Map the preparation pipeline's real boolean annotation to a class label."""
    return "HALLUCINATED" if bool(record["has_hallucination"]) else "GROUNDED"


def instruction_prompt(record: dict) -> str:
    """The supervised prefix. QLoRA masks loss here and predicts only the label."""
    return ("### Instruction\nDetermine whether the response is grounded only in the provided context. "
            "Return exactly GROUNDED or HALLUCINATED.\n\n"
            f"### Context\n{record['context']}\n\n### Response\n{record['response']}\n\n### Classification\n")


def load_prepared(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return [{"prompt": instruction_prompt(record), "label": label_from_record(record), "text": to_instruction(record)} for record in records]
