from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.graph import build_verification_graph
from src.config import get_settings
from src.models.llm import LlamaGenerator
from src.rag.retriever import EvidenceRetriever

app = FastAPI(title="AI Hallucination Detection & Verification", version="1.0.0")
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class AnalysisRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)


@app.get("/", include_in_schema=False)
def home(): return FileResponse(FRONTEND / "index.html")


@app.get("/health")
def health(): return {"status": "ok", "model": get_settings().model_name}


@app.post("/api/analyze")
def analyze(request: AnalysisRequest):
    settings = get_settings()
    try:
        retriever = EvidenceRetriever(settings)
        question_evidence = retriever.retrieve(request.prompt)
        answer = LlamaGenerator(settings).generate(request.prompt, [doc.page_content for doc in question_evidence])
        state = build_verification_graph(retriever).invoke({"question": request.prompt, "answer": answer, "claims": [], "retrievals": {}})
        return {"answer": answer, "claims": state["claims"], "hallucination_score": state["hallucination_score"],
                "confidence_score": state["confidence_score"], "classification": state["classification"],
                "generation_evidence": [{"text": doc.page_content, "source_id": doc.metadata.get("source_id")} for doc in question_evidence]}
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
