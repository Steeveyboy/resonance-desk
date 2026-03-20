"""Synthesizer — distils all agent responses into a single market signal.

Parses the debate result to produce:
- A numeric volatility score (0-100)
- A final market recommendation (BUY / HOLD / SHORT / SELL)
- A brief rationale
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from simulation.orchestrator import DebateResult


@dataclass
class SynthesisResult:
    """Final market signal synthesized from the multi-agent debate."""

    volatility_score: int
    recommendation: str  # BUY | HOLD | SHORT | SELL | NEUTRAL
    rationale: str
    stance_breakdown: dict[str, str]  # agent_name -> stance

    @property
    def color(self) -> str:
        """Return a Streamlit-friendly color string for the recommendation."""
        return {
            "BUY": "green",
            "HOLD": "orange",
            "SHORT": "red",
            "SELL": "red",
        }.get(self.recommendation, "gray")

    @property
    def emoji(self) -> str:
        """Return an emoji that represents the recommendation."""
        return {
            "BUY": "📈",
            "HOLD": "⏸️",
            "SHORT": "📉",
            "SELL": "🔻",
        }.get(self.recommendation, "❓")


class Synthesizer:
    """Converts a :class:`~simulation.orchestrator.DebateResult` into a
    :class:`SynthesisResult` by analysing stance votes and extracting a
    volatility score from the final risk-manager verdict.
    """

    _SCORE_PATTERN = re.compile(r"\b(\d{1,3})\s*/\s*100\b|\bscore[:\s]+(\d{1,3})\b", re.IGNORECASE)

    def synthesize(self, debate: DebateResult) -> SynthesisResult:
        """Synthesize the debate into a final market signal.

        Args:
            debate: A completed :class:`~simulation.orchestrator.DebateResult`.

        Returns:
            A :class:`SynthesisResult` with volatility score and recommendation.
        """
        stance_breakdown = self._collect_stances(debate)
        recommendation = self._majority_vote(stance_breakdown, debate)
        volatility_score = self._extract_volatility(debate)
        rationale = self._build_rationale(debate, recommendation, volatility_score)

        return SynthesisResult(
            volatility_score=volatility_score,
            recommendation=recommendation,
            rationale=rationale,
            stance_breakdown=stance_breakdown,
        )

    def _collect_stances(self, debate: DebateResult) -> dict[str, str]:
        stances: dict[str, str] = {}
        for r in debate.responses:
            stances[r.agent_name] = r.stance
        if debate.final_verdict:
            stances[debate.final_verdict.agent_name] = debate.final_verdict.stance
        return stances

    def _majority_vote(
        self, stances: dict[str, str], debate: DebateResult
    ) -> str:
        # The risk manager's verdict takes priority
        if debate.final_verdict and debate.final_verdict.stance not in ("", "NEUTRAL"):
            return debate.final_verdict.stance

        # Fall back to majority among all agents
        counts: dict[str, int] = {}
        for stance in stances.values():
            if stance:
                counts[stance] = counts.get(stance, 0) + 1
        if not counts:
            return "NEUTRAL"
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    def _extract_volatility(self, debate: DebateResult) -> int:
        """Extract the first numeric volatility score found in the verdict."""
        text = ""
        if debate.final_verdict:
            text = debate.final_verdict.response
        # Try combined agent text as fallback
        if not text:
            text = debate.transcript

        match = self._SCORE_PATTERN.search(text)
        if match:
            raw = match.group(1) or match.group(2)
            value = int(raw)
            return min(max(value, 0), 100)

        # Heuristic fallback based on stance distribution
        bearish = sum(
            1 for s in debate.responses if s.stance in ("SHORT", "SELL")
        )
        total = max(len(debate.responses), 1)
        return min(int((bearish / total) * 80) + 20, 100)

    def _build_rationale(
        self,
        debate: DebateResult,
        recommendation: str,
        volatility_score: int,
    ) -> str:
        agent_count = len(debate.responses)
        if debate.final_verdict:
            verdict_snippet = debate.final_verdict.response[:200].rstrip() + "…"
        else:
            verdict_snippet = "No final verdict available."

        return (
            f"{agent_count} agents debated the headline. "
            f"Consensus leans **{recommendation}** with a volatility score of **{volatility_score}/100**. "
            f"Risk manager summary: {verdict_snippet}"
        )
