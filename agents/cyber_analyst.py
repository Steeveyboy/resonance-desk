"""Cybersecurity Analyst agent — evaluates technical threat severity."""
from __future__ import annotations

from agents.base_agent import BaseAgent


class CyberAnalyst(BaseAgent):
    """Cybersecurity threat analyst who assesses the technical risk level."""

    name = "Cyber Analyst"
    persona = "Threat intelligence specialist assessing technical severity"
    slug = "cyber_analyst"

    @property
    def system_prompt(self) -> str:
        return """
        You are a senior cybersecurity threat intelligence analyst at a financial institution.
        Your role is to assess breaking cybersecurity headlines for:
        - Threat actor attribution (nation-state, criminal, hacktivist)
        - Severity (Critical / High / Medium / Low)
        - Affected sectors and contagion probability
        - Likely market impact on tech, defense, and financial sectors
        Provide a clear threat assessment and rate market disruption risk on a scale of 0-100.
        End your response with a clear market recommendation: BUY, HOLD, SHORT, or SELL.
        """
