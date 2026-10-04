"""Hugging Face Llama 3.1 generation; loaded only on demand."""
from __future__ import annotations

from src.config import Settings


class LlamaGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            device_map = "auto" if self.settings.device == "auto" else self.settings.device
            self._pipeline = pipeline("text-generation", model=self.settings.model_name,
                                      token=self.settings.hf_token, device_map=device_map)
        return self._pipeline

    def generate(self, question: str, evidence: list[str]) -> str:
        context = "\n\n".join(f"[{index + 1}] {text}" for index, text in enumerate(evidence))
        prompt = ("You are a careful assistant. Answer the question using only the provided evidence. "
                  "If the evidence is insufficient, say so clearly.\n\n"
                  f"Evidence:\n{context}\n\nQuestion: {question}\nAnswer:")
        output = self._load()(prompt, max_new_tokens=self.settings.max_new_tokens,
                              do_sample=False, return_full_text=False)
        return output[0]["generated_text"].strip()
