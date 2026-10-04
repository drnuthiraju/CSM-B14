from __future__ import annotations

class HallucinationAnalysisAgent:
    def run(self, state):
        # Severity: contradicted=1, unsupported=0.75, supported=0. The score is
        # an evidence-grounded claim proportion, not a calibrated probability.
        severity = {"supported": 0.0, "unsupported": 0.75, "contradicted": 1.0}
        claims = state["claims"]
        score = sum(severity[item["verdict"]] for item in claims) / max(1, len(claims))
        return {"hallucination_score": round(score, 3)}
