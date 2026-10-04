from __future__ import annotations

from src.config import Settings
from .vector_store import load_faiss


class EvidenceRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._store = None

    def retrieve(self, query: str):
        if self._store is None:
            self._store = load_faiss(self.settings)
        return self._store.similarity_search(query, k=self.settings.top_k)
