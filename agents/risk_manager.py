"""Risk Manager agent — moderates the debate and synthesizes agent views."""
from __future__ import annotations

from agents.base_agent import BaseAgent, AgentResponse


class RiskManager(BaseAgent):
    """Risk manager who balances all agent views and sets portfolio guardrails."""

    name = "Risk Manager"
    persona = "Chief Risk Officer balancing upside and downside scenarios"
    slug = "risk_manager"

    @property
    def agent_system_prompt(self) -> str:
        return """
        You are the Chief Risk Officer at a multi-strategy trading firm.
        You have heard arguments from bullish traders, bearish traders,
        cybersecurity analysts, and geopolitical analysts.
        Your job is to weigh all perspectives, apply risk management principles,
        and deliver a final, balanced assessment including:
        - An overall volatility score (0-100, where 100 = extreme volatility)
        - A final market recommendation: BUY, HOLD, SHORT, or SELL
        - Suggested position sizing (e.g., reduce leverage, hedge ratio)
        Be decisive and justify your conclusion.
        """

    def moderate(self, headline: str, debate_transcript: str) -> AgentResponse:  # noqa: F821
        """Produce a final ruling after seeing the full debate transcript.

        Args:
            headline: The original headline.
            debate_transcript: Concatenated responses from all other agents.

        Returns:
            An :class:`~agents.base_agent.AgentResponse` with the final verdict.
        """
        from utils.llm import call_llm_structured, StanceAnalysis

        user_message = (
            f"Breaking headline: {headline}\n\n"
            "--- Debate Transcript ---\n"
            f"{debate_transcript}\n"
            "--- End Transcript ---\n\n"
            "Based on all the above, provide your final risk assessment and recommendation."
        )
        raw: StanceAnalysis = call_llm_structured(
            system_prompt=self.system_prompt,
            user_message=user_message,
            agent_name=self.slug,
        )
        stance = raw.stance
        return AgentResponse(
            agent_name=self.name,
            persona=self.persona,
            response=raw.response,
            stance=stance,
        )
