from langchain_core.documents import Document

from src.agents.graph import build_verification_graph


class StubRetriever:
    def retrieve(self, query):
        return [Document(page_content="Albert Einstein developed the theory of relativity.", metadata={"source_id": "1", "source": "test"})]


def test_langgraph_returns_scored_claims():
    result = build_verification_graph(StubRetriever()).invoke({"question": "Who?", "answer": "Albert Einstein developed the theory of relativity.", "claims": [], "retrievals": {}})
    assert result["claims"][0]["verdict"] == "supported"
    assert result["hallucination_score"] == 0.0
