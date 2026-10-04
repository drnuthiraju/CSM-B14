"""Schema-aware loader for the official RAGTruth JSONL files."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

RESPONSE_REQUIRED = {"id", "source_id", "response", "labels", "split"}
SOURCE_REQUIRED = {"source_id", "source_info", "prompt", "task_type"}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing RAGTruth file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def load_ragtruth(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load and validate responses and source contexts without assuming extra columns."""
    response_rows = list(read_jsonl(dataset_dir / "response.jsonl"))
    source_rows = list(read_jsonl(dataset_dir / "source_info.jsonl"))
    if not response_rows or not source_rows:
        raise ValueError("RAGTruth response and source files must not be empty.")
    missing_response = RESPONSE_REQUIRED - response_rows[0].keys()
    missing_source = SOURCE_REQUIRED - source_rows[0].keys()
    if missing_response or missing_source:
        raise ValueError(f"Unexpected RAGTruth schema. Missing response={missing_response}, source={missing_source}")
    sources = {str(row["source_id"]): row for row in source_rows}
    absent_sources = {str(row["source_id"]) for row in response_rows} - sources.keys()
    if absent_sources:
        raise ValueError(f"Responses reference {len(absent_sources)} missing source records.")
    return response_rows, sources
