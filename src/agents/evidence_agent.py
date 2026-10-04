from __future__ import annotations

from src.rag.retriever import EvidenceRetriever


class EvidenceRetrievalAgent:
    def __init__(self, retriever: EvidenceRetriever) -> None:
        self.retriever = retriever

    def run(self, state):
        retrievals = {}
        for claim in state["claims"]:
            retrievals[claim["text"]] = [
                {"text": document.page_content, "source_id": document.metadata.get("source_id"), "source": document.metadata.get("source")}
                for document in self.retriever.retrieve(claim["text"])
            ]
        return {"retrievals": retrievals}
