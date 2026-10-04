from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.rag.ingest import ragtruth_documents
from src.rag.vector_store import build_faiss


def main() -> None:
    settings = get_settings()
    documents = ragtruth_documents(settings.ragtruth_path)
    build_faiss(documents, settings)
    print(f"Built FAISS index with {len(documents)} contexts at {settings.vector_store_path}")


if __name__ == "__main__":
    main()
