"""Leakage-aware source-group splitting."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def train_validation_split(records: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by source_id, never allowing the same RAG context into both sets."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["source_id"]].append(record)
    source_ids = np.array(sorted(grouped))
    rng = np.random.default_rng(seed)
    rng.shuffle(source_ids)
    cutoff = max(1, round(len(source_ids) * validation_fraction))
    validation_ids = set(source_ids[:cutoff])
    validation = [record for record in records if record["source_id"] in validation_ids]
    training = [record for record in records if record["source_id"] not in validation_ids]
    return training, validation
