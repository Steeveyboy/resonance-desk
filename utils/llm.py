"""LLM client wrapper.

Provides a thin wrapper around the OpenAI chat-completions API.
When ``OPENAI_API_KEY`` is not set the client falls back to *mock* mode,
returning deterministic placeholder responses so the app can be explored
without a live API key.
"""
from __future__ import annotations

import os
import textwrap
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_MOCK_RESPONSES: dict[str, str] = {
    "bull_trader": (
        "The specialists raise valid concerns, but I've seen this pattern before. "
        "Despite the cyber and geopolitical risk flags, underlying fundamentals remain strong. "
        "Tech stocks have historically recovered quickly from these scares. "
        "I recommend a cautious BUY on dips — volatility is a buying opportunity."
    ),
    "bear_trader": (
        "The specialist assessments confirm my thesis: this headline signals systemic risk. "
        "The cyber analyst's CRITICAL severity rating and the geopolitical knock-ons the "
        "geopolitical analyst described make a SHORT position compelling. "
        "Supply chain disruptions and regulatory backlash are likely — "
        "expect 15-20% downside pressure over the next two weeks."
    ),
    "cyber_analyst": (
        "Threat actor fingerprints suggest a nation-state APT campaign. "
        "Critical infrastructure exposure is high. I assess this as a CRITICAL severity "
        "event with a contagion probability of 65%. Markets should price in prolonged disruption."
    ),
    "geopolitical_analyst": (
        "The geopolitical context elevates this beyond a technical incident. "
        "Sanctions risk, diplomatic fallout, and energy market knock-ons are plausible. "
        "Regional volatility index likely to spike 20-30 points within 48 hours."
    ),
    "risk_manager": (
        "Given the divergent views, prudent risk management calls for a HOLD with "
        "strict stop-losses. Reduce leverage by 50%. "
        "Volatility score: 72/100. Recommended action: HOLD."
    ),
}


def _mock_response(agent_name: str, headline: str) -> str:
    base = _MOCK_RESPONSES.get(agent_name, "Insufficient data to form a view.")
    return f"[MOCK — no API key] {base}"


def call_llm(
    system_prompt: str,
    user_message: str,
    agent_name: str = "agent",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Call the LLM and return the assistant reply as a string.

    Falls back to a mock response when ``OPENAI_API_KEY`` is absent.

    Args:
        system_prompt: Role/persona instructions for the agent.
        user_message: The headline or prompt passed to the agent.
        agent_name: Key used to look up a mock response (matches agent slugs).
        model: OpenAI model identifier; defaults to the ``OPENAI_MODEL`` env var
               or ``gpt-4o-mini``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the completion.

    Returns:
        The assistant reply text.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-...":
        return _mock_response(agent_name, user_message)

    try:
        from openai import OpenAI  # imported lazily so mock mode has no hard dep

        client = OpenAI(api_key=api_key)
        resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": textwrap.dedent(system_prompt)},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return f"[LLM error: {exc}]"
