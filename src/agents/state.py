from __future__ import annotations

from typing import NotRequired, TypedDict


class Claim(TypedDict):
    text: str
    verdict: NotRequired[str]
    confidence: NotRequired[float]
    explanation: NotRequired[str]
    evidence: NotRequired[list[dict]]


class VerificationState(TypedDict):
    question: str
    answer: str
    claims: list[Claim]
    retrievals: dict[str, list[dict]]
    hallucination_score: NotRequired[float]
    confidence_score: NotRequired[float]
    classification: NotRequired[str]
