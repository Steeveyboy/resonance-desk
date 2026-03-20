"""Resonance — Multi-agent market simulation.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from simulation.orchestrator import DebateOrchestrator
from simulation.synthesizer import Synthesizer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resonance — Market Intelligence",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📡 Resonance")
st.caption(
    "Multi-agent simulation — feed a breaking headline and watch five AI agents "
    "debate the market impact in real time."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(
        "Set **OPENAI_API_KEY** in a `.env` file to use live AI responses. "
        "Without a key the app runs in **mock mode** with placeholder answers.",
        icon="ℹ️",
    )
    st.markdown("**Agents in this simulation**")
    agents_info = [
        ("🔬", "Cyber Analyst", "Assesses technical threat severity"),
        ("🌍", "Geopolitical Analyst", "Evaluates macro & diplomatic risk"),
        ("📈", "Bull Trader", "Looks for buying opportunities"),
        ("📉", "Bear Trader", "Looks for short/sell opportunities"),
        ("🛡️", "Risk Manager", "Moderates debate & delivers verdict"),
    ]
    for icon, name, desc in agents_info:
        st.markdown(f"{icon} **{name}** — {desc}")

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.subheader("📰 Breaking Headline")

example_headlines = [
    "Select an example…",
    "Major cyberattack cripples US East Coast power grid, attributed to nation-state actor",
    "North Korea launches ballistic missile over Japan, South Korea raises alert level",
    "Critical zero-day in OpenSSL exposes banking infrastructure worldwide",
    "Russia blocks European gas pipelines amid NATO expansion talks",
    "China seizes Taiwan Strait shipping lanes; US carrier group deployed",
]

col_input, col_example = st.columns([3, 2])
with col_example:
    selected_example = st.selectbox("Or pick an example:", example_headlines)
with col_input:
    default_headline = (
        "" if selected_example == example_headlines[0] else selected_example
    )
    headline = st.text_area(
        "Enter a cybersecurity or geopolitical headline:",
        value=default_headline,
        height=100,
        placeholder="e.g. 'Critical zero-day in banking sector SWIFT network…'",
    )

run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
if run_button:
    if not headline.strip():
        st.warning("Please enter a headline before running the simulation.")
        st.stop()

    st.divider()
    st.subheader("🤖 Agent Debate")

    orchestrator = DebateOrchestrator()
    synthesizer = Synthesizer()

    agent_icons = {
        "Cyber Analyst": "🔬",
        "Geopolitical Analyst": "🌍",
        "Bull Trader": "📈",
        "Bear Trader": "📉",
        "Risk Manager": "🛡️",
    }
    stance_colors = {
        "BUY": "🟢",
        "HOLD": "🟡",
        "SHORT": "🔴",
        "SELL": "🔴",
        "NEUTRAL": "⚪",
    }

    debate_result = None
    responses_collected = []
    progress = st.progress(0, text="Starting debate…")
    total_agents = 5  # 4 debate agents + 1 risk manager

    # Stream responses and render each one as it arrives
    for i, response in enumerate(orchestrator.stream(headline)):
        pct = int((i + 1) / total_agents * 100)
        progress.progress(pct, text=f"Waiting for {response.agent_name}…")
        responses_collected.append(response)

        icon = agent_icons.get(response.agent_name, "🤖")
        stance_dot = stance_colors.get(response.stance, "⚪")
        is_verdict = response.agent_name == "Risk Manager"

        with st.expander(
            f"{icon} **{response.agent_name}** — {response.persona}  {stance_dot} *{response.stance}*",
            expanded=is_verdict,
        ):
            st.markdown(response.response)

    progress.empty()

    # Rebuild DebateResult for synthesizer
    from simulation.orchestrator import DebateResult
    debate_result = DebateResult(headline=headline)
    debate_result.responses = responses_collected[:-1]  # all except risk manager
    debate_result.final_verdict = responses_collected[-1]

    # ---------------------------------------------------------------------------
    # Synthesis panel
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Synthesized Market Signal")

    synthesis = synthesizer.synthesize(debate_result)

    col_score, col_rec, col_breakdown = st.columns([1, 1, 2])

    with col_score:
        st.metric(
            label="🌡️ Volatility Score",
            value=f"{synthesis.volatility_score} / 100",
            delta=None,
            help="0 = calm markets, 100 = extreme volatility expected",
        )
        vol = synthesis.volatility_score
        if vol >= 70:
            st.error("⚠️ High volatility regime")
        elif vol >= 40:
            st.warning("🔶 Elevated volatility")
        else:
            st.success("✅ Low volatility")

    with col_rec:
        color_map = {"BUY": "green", "HOLD": "orange", "SHORT": "red", "SELL": "red"}
        color = color_map.get(synthesis.recommendation, "gray")
        st.markdown(
            f"<h1 style='color:{color};text-align:center'>"
            f"{synthesis.emoji} {synthesis.recommendation}</h1>",
            unsafe_allow_html=True,
        )

    with col_breakdown:
        st.markdown("**Stance breakdown by agent**")
        for agent_name, stance in synthesis.stance_breakdown.items():
            dot = stance_colors.get(stance, "⚪")
            icon = agent_icons.get(agent_name, "🤖")
            st.markdown(f"{icon} {agent_name}: {dot} **{stance}**")

    st.info(synthesis.rationale)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Resonance v0.1 — skeleton app — not financial advice.")
