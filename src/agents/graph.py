from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.claim_agent import ClaimExtractionAgent
from src.agents.evidence_agent import EvidenceRetrievalAgent
from src.agents.hallucination_agent import HallucinationAnalysisAgent
from src.agents.judge_agent import FinalJudgeAgent
from src.agents.state import VerificationState
from src.agents.verification_agent import ClaimVerificationAgent
from src.rag.retriever import EvidenceRetriever


def build_verification_graph(retriever: EvidenceRetriever):
    """Create explicit, typed agent transitions for auditability."""
    graph = StateGraph(VerificationState)
    graph.add_node("claim_extraction", ClaimExtractionAgent().run)
    graph.add_node("evidence_retrieval", EvidenceRetrievalAgent(retriever).run)
    graph.add_node("claim_verification", ClaimVerificationAgent().run)
    graph.add_node("hallucination_analysis", HallucinationAnalysisAgent().run)
    graph.add_node("final_judge", FinalJudgeAgent().run)
    graph.add_edge(START, "claim_extraction")
    graph.add_edge("claim_extraction", "evidence_retrieval")
    graph.add_edge("evidence_retrieval", "claim_verification")
    graph.add_edge("claim_verification", "hallucination_analysis")
    graph.add_edge("hallucination_analysis", "final_judge")
    graph.add_edge("final_judge", END)
    return graph.compile()
