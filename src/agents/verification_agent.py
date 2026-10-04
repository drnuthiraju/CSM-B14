from __future__ import annotations

from src.models.detector import grounding_overlap, has_opposing_polarity


class ClaimVerificationAgent:
    """Evidence-only verifier with deterministic, inspectable decision criteria."""
    def run(self, state):
        verified = []
        for claim in state["claims"]:
            evidence = state["retrievals"].get(claim["text"], [])
            best = evidence[0] if evidence else None
            overlap = grounding_overlap(claim["text"], best["text"]) if best else 0.0
            if best and overlap >= 0.60 and has_opposing_polarity(claim["text"], best["text"]):
                verdict, reason = "contradicted", "Relevant evidence has opposite negation polarity."
            elif best and overlap >= 0.60:
                verdict, reason = "supported", "Most claim terms are present in retrieved evidence."
            else:
                verdict, reason = "unsupported", "Retrieved evidence does not sufficiently ground the claim."
            verified.append({**claim, "verdict": verdict, "confidence": round(overlap, 3), "explanation": reason, "evidence": evidence})
        return {"claims": verified}
