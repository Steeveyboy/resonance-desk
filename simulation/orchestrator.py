"""Debate orchestrator.

Coordinates all agents in sequence: specialist analysts speak first,
traders respond, and finally the risk manager delivers a verdict.

If any agent fails, the debate stops there: later rounds build on earlier
responses, so continuing would spend tokens on analysis of an error message.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Generator

from agents.base_agent import AgentResponse, BaseAgent
from agents.bull_trader import BullTrader
from agents.bear_trader import BearTrader
from agents.cyber_analyst import CyberAnalyst
from agents.geopolitical_analyst import GeopoliticalAnalyst
from agents.risk_manager import RiskManager

logger = logging.getLogger(__name__)


def _build_transcript(responses: list[AgentResponse]) -> str:
    """Format a list of agent responses into a readable transcript string."""
    lines = []
    for r in responses:
        lines.append(f"[{r.agent_name}] ({r.persona})\n{r.response}")
    return "\n\n".join(lines)


@dataclass
class DebateResult:
    """Full output of a single debate run."""
    headline: str
    responses: list[AgentResponse] = field(default_factory=list)
    final_verdict: AgentResponse | None = None

    @property
    def transcript(self) -> str:
        """Return a text transcript of all agent responses (excluding final verdict)."""
        return _build_transcript(self.responses)

    @property
    def failed_response(self) -> AgentResponse | None:
        """Return the response that stopped the debate, if any agent failed."""
        for r in [*self.responses, self.final_verdict]:
            if r is not None and r.failed:
                return r
        return None

    @property
    def aborted(self) -> bool:
        """Return whether the debate stopped early because an agent failed."""
        return self.failed_response is not None


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
            A :class:`DebateResult` containing all responses and the final
            verdict, or a partial result if an agent failed.
        """
        debate = self.stream(headline)
        while True:
            try:
                next(debate)
            except StopIteration as stop:
                return stop.value

    def stream(
        self,
        headline: str,
        on_response: Callable[[AgentResponse], None] | None = None,
    ) -> Generator[AgentResponse, None, DebateResult]:
        """Stream agent responses one at a time.

        Yields each :class:`AgentResponse` as it is produced.
        Optionally calls *on_response* callback for each response.
        Returns the final :class:`DebateResult` as the generator's return value.

        If an agent fails, its error response is yielded and the generator
        stops without calling any further agents; the returned result then has
        :attr:`DebateResult.aborted` set.

        Args:
            headline: The breaking news headline to simulate.
            on_response: Optional callback invoked after each response.

        Yields:
            :class:`AgentResponse` objects in debate order.

        Returns:
            The completed (or aborted) :class:`DebateResult`.
        """
        result = DebateResult(headline=headline)
        started = time.perf_counter()
        logger.info("Debate started: %r", headline)

        def emit(response: AgentResponse) -> bool:
            """Notify the callback; return False if the debate must stop."""
            if on_response:
                on_response(response)
            if response.failed:
                logger.error(
                    "Stopping debate after %.1fs: %s failed (%s)",
                    time.perf_counter() - started,
                    response.agent_name,
                    response.error,
                )
                return False
            return True

        # Round 1 — analysts speak (headline only)
        logger.info("Round 1: analysts")
        for agent in self._analysts:
            response = agent.analyze(headline)
            result.responses.append(response)
            keep_going = emit(response)
            yield response
            if not keep_going:
                return result

        # Round 2 — traders react, informed by the specialist assessments
        logger.info("Round 2: traders")
        analyst_transcript = _build_transcript(result.responses)
        for agent in self._traders:
            response = agent.analyze(headline, context=analyst_transcript)
            result.responses.append(response)
            keep_going = emit(response)
            yield response
            if not keep_going:
                return result

        # Final round — risk manager synthesizes everything
        logger.info("Round 3: risk manager")
        verdict = self._risk_manager.moderate(headline, result.transcript)
        result.final_verdict = verdict
        keep_going = emit(verdict)
        yield verdict
        if keep_going:
            logger.info(
                "Debate finished in %.1fs, verdict=%s",
                time.perf_counter() - started,
                verdict.stance,
            )
        return result
