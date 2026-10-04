from __future__ import annotations

import re


class ClaimExtractionAgent:
    """Extract sentence-level declarative claims; questions are not claims."""
    def run(self, state):
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", state["answer"]) if item.strip()]
        return {"claims": [{"text": sentence} for sentence in sentences if not sentence.endswith("?")]}
