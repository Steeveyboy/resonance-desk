"""Debate orchestrator.

Coordinates all agents in sequence: specialist analysts speak first,
traders respond, and finally the risk manager delivers a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generator

from agents.base_agent import AgentResponse, BaseAgent
from agents.bull_trader import BullTrader
from agents.bear_trader import BearTrader
from agents.cyber_analyst import CyberAnalyst
from agents.geopolitical_analyst import GeopoliticalAnalyst
from agents.risk_manager import RiskManager


@dataclass
class DebateResult:
    """Full output of a single debate run."""

    headline: str
    responses: list[AgentResponse] = field(default_factory=list)
    final_verdict: AgentResponse | None = None

    @property
    def transcript(self) -> str:
        """Return a text transcript of all agent responses (excluding final verdict)."""
        lines = []
        for r in self.responses:
            lines.append(f"[{r.agent_name}] ({r.persona})\n{r.response}\n")
        return "\n".join(lines)


class DebateOrchestrator:
    """Runs the multi-agent debate for a given headline.

    Usage::

        orchestrator = DebateOrchestrator()
        result = orchestrator.run("Major cyberattack hits US power grid")

    You may also stream responses one-by-one using :meth:`stream`::

        for response in orchestrator.stream(headline):
            print(response.agent_name, response.response)
    """

    def __init__(self) -> None:
        self._analysts: list[BaseAgent] = [CyberAnalyst(), GeopoliticalAnalyst()]
        self._traders: list[BaseAgent] = [BullTrader(), BearTrader()]
        self._risk_manager = RiskManager()

    def run(self, headline: str) -> DebateResult:
        """Run the full debate synchronously and return a :class:`DebateResult`.

        Args:
            headline: The breaking news headline to simulate.

        Returns:
            A :class:`DebateResult` containing all responses and the final verdict.
        """
        result = DebateResult(headline=headline)

        # Round 1 — specialist analysts assess the situation
        for agent in self._analysts:
            result.responses.append(agent.analyze(headline))

        # Round 2 — traders react to the specialist assessments
        for agent in self._traders:
            result.responses.append(agent.analyze(headline))

        # Final round — risk manager synthesizes everything
        result.final_verdict = self._risk_manager.moderate(
            headline, result.transcript
        )
        return result

    def stream(
        self,
        headline: str,
        on_response: Callable[[AgentResponse], None] | None = None,
    ) -> Generator[AgentResponse, None, DebateResult]:
        """Stream agent responses one at a time.

        Yields each :class:`AgentResponse` as it is produced.
        Optionally calls *on_response* callback for each response.
        Returns the final :class:`DebateResult` as the generator's return value.

        Args:
            headline: The breaking news headline to simulate.
            on_response: Optional callback invoked after each response.

        Yields:
            :class:`AgentResponse` objects in debate order.

        Returns:
            The completed :class:`DebateResult`.
        """
        result = DebateResult(headline=headline)

        for agent in self._analysts + self._traders:
            response = agent.analyze(headline)
            result.responses.append(response)
            if on_response:
                on_response(response)
            yield response

        verdict = self._risk_manager.moderate(headline, result.transcript)
        result.final_verdict = verdict
        if on_response:
            on_response(verdict)
        yield verdict

        return result
