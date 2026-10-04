from __future__ import annotations

from src.config import Settings


def create_embeddings(settings: Settings):
    """Create the configured local Hugging Face embedding adapter on demand."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
