"""Prepare RAGTruth records and print verifiable corpus statistics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.data.preprocess import dataset_statistics, normalize_ragtruth
from src.data.split import train_validation_split


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=get_settings().ragtruth_path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ragtruth"))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=get_settings().seed)
    args = parser.parse_args()
    records = normalize_ragtruth(args.dataset_dir)
    official_train = [row for row in records if row["split"] == "train"]
    official_test = [row for row in records if row["split"] == "test"]
    train, validation = train_validation_split(official_train, args.validation_fraction, args.seed)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "test.jsonl", official_test)
    stats = dataset_statistics(records) | {"prepared_train": len(train), "prepared_validation": len(validation), "prepared_test": len(official_test), "seed": args.seed}
    (args.output_dir / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
