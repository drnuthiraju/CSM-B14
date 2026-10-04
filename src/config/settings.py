"""Central, environment-based configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_name: str
    embedding_model: str
    ragtruth_path: Path
    vector_store_path: Path
    hf_token: str | None
    top_k: int
    max_new_tokens: int
    device: str
    seed: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read configuration once; paths remain project-relative by default."""
    return Settings(
        model_name=os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        ragtruth_path=Path(os.getenv("RAGTRUTH_PATH", "data/RAGTruth/dataset")),
        vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store/ragtruth_faiss")),
        hf_token=os.getenv("HF_TOKEN") or None,
        top_k=int(os.getenv("TOP_K", "4")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
        device=os.getenv("DEVICE", "auto"),
        seed=int(os.getenv("SEED", "42")),
    )
