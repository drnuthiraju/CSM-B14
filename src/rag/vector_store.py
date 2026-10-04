from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.config import Settings
from src.models.embeddings import create_embeddings


def build_faiss(documents, settings: Settings) -> None:
    if not documents:
        raise ValueError("Cannot create a vector index from zero documents.")
    store = FAISS.from_documents(documents, create_embeddings(settings))
    settings.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.vector_store_path))


def load_faiss(settings: Settings):
    path = settings.vector_store_path
    if not Path(path).exists():
        raise FileNotFoundError(f"No vector store at {path}. Run: python scripts/build_index.py")
    # FAISS stores pickle metadata locally; this project only loads its own locally built index.
    return FAISS.load_local(str(path), create_embeddings(settings), allow_dangerous_deserialization=True)
