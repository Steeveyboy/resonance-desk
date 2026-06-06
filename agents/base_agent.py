"""Base agent abstraction.

All market-participant agents inherit from :class:`BaseAgent`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils.llm import StanceEnum, StanceAnalysis, call_llm_structured


@dataclass
class AgentResponse:
    """Structured response produced by an agent during the debate."""

    agent_name: str
    persona: str
    response: str
    stance: StanceEnum = StanceEnum.NEUTRAL  # e.g. "BUY", "SELL", "SHORT", "HOLD"


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
    def _base_system_prompt(self) -> str:
        """Common instructions appended to every agent's system prompt."""
        return """
        Format your response using markdown:
        - Use **bold** for key terms and recommendations
        - Use bullet points for lists of risks or factors
        - End with a clear recommendation on its own line: **Recommendation: BUY / SELL / SHORT / HOLD**
        """

    @property
    @abstractmethod
    def agent_system_prompt(self) -> str:
        """Return the agent-specific system prompt describing its role and perspective."""

    @property
    def system_prompt(self) -> str:
        """Return the system-level persona prompt for this agent."""
        return f"{self._base_system_prompt}\n\n{self.agent_system_prompt}"
    

    def analyze(self, headline: str, context: str = "") -> AgentResponse:
        """Analyze *headline* and return a structured :class:`AgentResponse`.

        Args:
            headline: The breaking news headline to evaluate.
            context: Optional prior analysis from earlier agents in the debate,
                     shown to this agent before it responds.

        Returns:
            An :class:`AgentResponse` with the agent's viewpoint.
        """
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
        result: StanceAnalysis = call_llm_structured(
            system_prompt=self.system_prompt,
            user_message=user_message,
            agent_name=self.slug,
        )
        return AgentResponse(
            agent_name=self.name,
            persona=self.persona,
            response=result.response,
            stance=result.stance,
        )
