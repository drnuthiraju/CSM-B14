from pathlib import Path

from src.data.preprocess import normalize_ragtruth
from src.data.split import train_validation_split


def test_ragtruth_records_preserve_annotations():
    records = normalize_ragtruth(Path("data/RAGTruth/dataset"))
    assert len(records) == 17790
    assert {"context", "response", "spans", "has_hallucination"} <= records[0].keys()


def test_source_groups_do_not_leak():
    rows = [{"source_id": str(i // 2)} for i in range(20)]
    train, validation = train_validation_split(rows, 0.2, 42)
    assert {row["source_id"] for row in train}.isdisjoint({row["source_id"] for row in validation})
