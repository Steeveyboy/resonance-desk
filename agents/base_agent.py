"""Base agent abstraction.

All market-participant agents inherit from :class:`BaseAgent`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Structured response produced by an agent during the debate."""

    agent_name: str
    persona: str
    response: str
    stance: str = ""  # e.g. "BUY", "SELL", "SHORT", "HOLD"


class BaseAgent(ABC):
    """Abstract base class for all debate agents.

    Subclasses must implement :meth:`system_prompt` and :meth:`analyze`.
    """

    #: Human-readable display name shown in the UI.
    name: str = "Agent"
    #: Short one-line description of the persona.
    persona: str = "Market participant"
    #: Slug used for mock-response lookup (matches keys in ``utils.llm._MOCK_RESPONSES``).
    slug: str = "agent"

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system-level persona prompt for this agent."""

    def analyze(self, headline: str, context: str = "") -> AgentResponse:
        """Analyze *headline* and return a structured :class:`AgentResponse`.

        Args:
            headline: The breaking news headline to evaluate.
            context: Optional prior analysis from earlier agents in the debate,
                     shown to this agent before it responds.

        Returns:
            An :class:`AgentResponse` with the agent's viewpoint.
        """
        from utils.llm import call_llm

        if context:
            user_message = (
                f"Breaking headline: {headline}\n\n"
                "--- Specialist Analysis ---\n"
                f"{context}\n"
                "--- End Specialist Analysis ---\n\n"
                "Given the specialist assessments above, provide your market "
                "stance in 3-5 sentences."
            )
        else:
            user_message = (
                f"Breaking headline: {headline}\n\n"
                "Provide your analysis and market stance in 3-5 sentences."
            )
        raw = call_llm(
            system_prompt=self.system_prompt,
            user_message=user_message,
            agent_name=self.slug,
        )
        stance = self._extract_stance(raw)
        return AgentResponse(
            agent_name=self.name,
            persona=self.persona,
            response=raw,
            stance=stance,
        )

    @staticmethod
    def _extract_stance(text: str) -> str:
        """Heuristically extract a BUY / SHORT / HOLD stance from free text."""
        upper = text.upper()
        for keyword in ("SHORT", "SELL", "BUY", "HOLD"):
            if keyword in upper:
                return keyword
        return "NEUTRAL"
