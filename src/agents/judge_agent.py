from __future__ import annotations

class FinalJudgeAgent:
    def run(self, state):
        claims = state["claims"]
        # Confidence is the average evidence overlap; it is explicitly not calibrated.
        confidence = sum(claim["confidence"] for claim in claims) / max(1, len(claims))
        score = state["hallucination_score"]
        label = "mostly supported" if score < 0.25 else "mixed evidence" if score < 0.60 else "potential hallucination"
        return {"confidence_score": round(confidence, 3), "classification": label}
