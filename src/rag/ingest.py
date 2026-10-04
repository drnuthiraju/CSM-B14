from __future__ import annotations

from langchain_core.documents import Document

from src.data.preprocess import normalize_ragtruth


def ragtruth_documents(dataset_dir):
    """Create one metadata-rich document per unique RAGTruth source context."""
    seen: set[str] = set()
    documents: list[Document] = []
    for record in normalize_ragtruth(dataset_dir):
        if record["source_id"] not in seen:
            seen.add(record["source_id"])
            documents.append(Document(page_content=record["context"], metadata={
                "source_id": record["source_id"], "task_type": record["task_type"], "source": "RAGTruth",
            }))
    return documents
