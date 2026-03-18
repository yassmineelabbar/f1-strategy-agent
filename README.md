# 🏎️ F1 Strategy Agent

An AI agent that analyses Formula 1 race situations and gives real-time pit stop strategy recommendations — powered by **Llama 3.3 70B** (via Groq) and real **FastF1** telemetry data.

## What it does

Ask it a race situation in plain English:

> *"Lap 42 of 58 at Monaco 2024. Verstappen is P1 on Hards (21 laps old). Norris is 4.2s behind on Mediums. Should Norris pit for an undercut?"*

The agent autonomously:
1. Fetches real lap-by-lap tire data from FastF1
2. Checks race gaps and positions
3. Runs undercut/overcut window math
4. Optionally checks circuit weather
5. Returns a structured recommendation like a real pit wall engineer

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Agent framework | Custom tool-calling loop |
| F1 data | FastF1 + Open-Meteo (weather) |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your Groq API key in the sidebar (get a free key at [console.groq.com](https://console.groq.com)).

## Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. In Advanced Settings → Secrets, add:
```toml
GROQ_API_KEY = "your_key_here"
```
4. Deploy — you'll get a public URL in ~2 minutes

## Project structure

```
f1-strategy-agent/
├── app.py            # Streamlit UI
├── agent.py          # Agent loop + tool functions
└── requirements.txt  # Dependencies
```
