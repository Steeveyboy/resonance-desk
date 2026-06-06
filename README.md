# Resonance 📡

> **A multi-agent market intelligence sandbox** for exploring how cybersecurity and geopolitical headlines can ripple into sentiment, volatility, and trading posture.

Resonance turns a breaking headline into a live AI debate between a cyber analyst, a geopolitical analyst, a bull trader, a bear trader, and a risk manager. The goal is simple: **make complex risk events easier to interpret** by surfacing competing viewpoints and distilling them into a single market signal.

## Why Resonance exists 🌍

Cyber incidents and geopolitical shocks rarely move markets in a straight line. Resonance is built to explore that ambiguity by letting multiple personas react to the same event:

- 🔬 **Specialists** assess technical and macro risk
- 📈📉 **Traders** argue bullish and bearish interpretations
- 🛡️ **A risk manager** synthesizes the debate into a recommendation and volatility score

The app supports:

- ⚡ **Live mode** with OpenAI-backed responses
- 🧪 **Mock mode** with deterministic placeholder outputs when no API key is set

## Tech stack 🧰

| Layer | Tools |
| --- | --- |
| UI | Streamlit |
| LLM integration | OpenAI Python SDK, Instructor |
| Data validation | Pydantic |
| Config | python-dotenv |
| Language | Python |

## Getting started 🚀

### 1. Clone the repo 📥

```bash
git clone https://github.com/<your-org-or-user>/resonance-desk.git
cd resonance-desk
```

### 2. Create and activate a virtual environment 🐍

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies 📦

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables 🔐

Create a `.env` file in the repo root.

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.4-mini
```

`OPENAI_API_KEY` is optional. If it is missing, the app starts in **mock mode**, which is useful for UI development, demos, and local exploration.

### 5. Run the app ▶️

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal, enter a headline, and run the simulation.

## How it works 🧠

1. **Cyber Analyst** evaluates technical severity and contagion risk.
2. **Geopolitical Analyst** evaluates macro, diplomatic, and sanctions impact.
3. **Bull Trader** argues the upside case.
4. **Bear Trader** argues the downside case.
5. **Risk Manager** issues the final recommendation and volatility score.

The synthesizer then combines the debate into a single market signal shown in the UI.

## What you get ✨

- 📰 Headline-driven simulations for cyber and geopolitical events
- 🤖 Structured multi-agent debate with distinct market personas
- 🌡️ A synthesized volatility score and final recommendation
- 🧩 Mock-mode fallbacks so the app remains usable without API access

## Repository layout 🗂️

```text
.
├── app.py                  # Streamlit entrypoint
├── agents/                 # Agent personas and debate roles
├── simulation/             # Orchestration and synthesis logic
├── utils/                  # Shared LLM utilities
├── requirements.txt
└── testing_instructor.py   # Small instructor experiment script
```

## Contributing 🤝

Contributions are welcome, especially around:

- richer agent behavior and prompts
- better synthesis and scoring logic
- improved UI/UX in Streamlit
- evaluation, testing, and reproducibility
- additional market or risk personas

To contribute:

1. Fork the repository and create a feature branch.
2. Make focused changes with clear commit messages.
3. Run the app locally and confirm the affected flow still works.
4. Open a pull request with a concise description of the change and, for UI updates, screenshots if helpful.

When possible, keep contributions small and composable so simulation behavior is easy to review.

## Notes ⚠️

- 🧭 This project is an exploration and prototyping tool.
- 💸 It is **not financial advice**.
