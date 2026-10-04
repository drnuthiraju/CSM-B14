"""Convert RAGTruth annotations to reproducible response-level training records."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .loader import load_ragtruth


def source_to_text(source_info: Any) -> str:
    """Serialize the source faithfully whether RAGTruth supplies prose or JSON."""
    return source_info if isinstance(source_info, str) else json.dumps(source_info, ensure_ascii=False, sort_keys=True)


def normalize_ragtruth(dataset_dir: Path) -> list[dict[str, Any]]:
    """Preserve spans while deriving a groundedness label from real annotations.

    A response is labelled `has_hallucination` exactly when RAGTruth provides one
    or more annotated spans. `implicit_true` stays present in metadata: it is
    true externally but unsupported by the supplied RAG context, which matters
    for the project's evidence-grounded verification task.
    """
    responses, sources = load_ragtruth(dataset_dir)
    records: list[dict[str, Any]] = []
    for row in responses:
        source = sources[str(row["source_id"])]
        spans = row["labels"]
        records.append({
            "id": str(row["id"]), "source_id": str(row["source_id"]),
            "split": row["split"], "task_type": source["task_type"], "model": row.get("model"),
            "prompt": source["prompt"], "context": source_to_text(source["source_info"]),
            "response": row["response"], "has_hallucination": bool(spans), "spans": spans,
            "quality": row.get("quality", "unknown"),
        })
    return records


def dataset_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "hallucinated_responses": sum(record["has_hallucination"] for record in records),
        "hallucination_spans": sum(len(record["spans"]) for record in records),
        "splits": dict(Counter(record["split"] for record in records)),
        "task_types": dict(Counter(record["task_type"] for record in records)),
    }
