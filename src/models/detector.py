"""Transparent lexical grounding helpers used by the verification agent."""
from __future__ import annotations

import re

TOKEN = re.compile(r"[A-Za-z0-9]+")
NEGATIONS = {"no", "not", "never", "none", "without", "false"}


def token_set(text: str) -> set[str]:
    return {item.lower() for item in TOKEN.findall(text) if len(item) > 1}


def grounding_overlap(claim: str, evidence: str) -> float:
    claim_tokens = token_set(claim)
    return len(claim_tokens & token_set(evidence)) / max(1, len(claim_tokens))


def has_opposing_polarity(claim: str, evidence: str) -> bool:
    return bool(token_set(claim) & NEGATIONS) != bool(token_set(evidence) & NEGATIONS)
