import streamlit as st
from agent import run_strategy_agent

st.set_page_config(
    page_title="F1 Strategy Agent",
    page_icon="🏎️",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏎️ F1 Strategy Agent")
st.caption("Powered by Llama 3.3 via Groq + FastF1 data")

# ── API key ────────────────────────────────────────────────────────────────────
# On Streamlit Cloud: loaded from st.secrets (never visible in code).
# Locally: falls back to a sidebar input for easy testing.
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = st.sidebar.text_input(
        "Groq API key",
        type="password",
        help="Get a free key at console.groq.com"
    )

# ── Sidebar presets ────────────────────────────────────────────────────────────
st.sidebar.header("Preset scenarios")
st.sidebar.caption("Click one to load it, or write your own below.")

presets = {
    "Monaco 2024 — Norris undercut?": """It's lap 42 of 58 at Monaco 2024.
Verstappen is on Hard tires (21 laps old), running P1.
Norris is 4.2 seconds behind on Medium tires (18 laps old), running P2.
Should Norris pit for an undercut attempt?""",

    "Silverstone 2024 — Hamilton in rain": """Lap 28 of 52 at Silverstone 2024.
Hamilton is P3 on Medium tires, 18 laps old.
The car ahead (Leclerc) is 6.8 seconds up the road.
There is a 40% chance of rain in the next 30 minutes.
What should Hamilton do?""",

    "Monza 2024 — Late race overcut": """Lap 47 of 53 at Monza 2024.
Sainz is P4 on Hard tires, 28 laps old, showing heavy degradation.
The car ahead is 3.1 seconds up the road.
Is there any point pitting this late, or should he stay out?""",
}

selected = None
for label in presets:
    if st.sidebar.button(label, use_container_width=True):
        selected = presets[label]

# ── Question input ─────────────────────────────────────────────────────────────
question = st.text_area(
    "Describe the race situation",
    value=selected if selected else "",
    height=160,
    placeholder="e.g. Lap 30 of 57 at Spa 2024. Verstappen is P1 on Hard tires age 22 laps...",
)

# ── Run ────────────────────────────────────────────────────────────────────────
run_disabled = not api_key or not question.strip()
if st.button("Analyse strategy", type="primary", disabled=run_disabled):

    if not api_key:
        st.warning("Add your Groq API key in the sidebar first.")
        st.stop()

    st.divider()

    with st.expander("Agent reasoning (tool calls)", expanded=False):
        tool_log = st.empty()
        log_lines = []

    result_placeholder = st.empty()

    with st.spinner("Agent thinking..."):
        for event_type, content in run_strategy_agent(question, api_key):

            if event_type == 'tool_call':
                log_lines.append(f"**Calling:** `{content}`")
                tool_log.markdown("\n\n".join(log_lines))

            elif event_type == 'tool_result':
                log_lines.append(f"```\n{content[:300]}\n```")
                tool_log.markdown("\n\n".join(log_lines))

            elif event_type == 'answer':
                result_placeholder.markdown("### Recommendation")
                result_placeholder.markdown(content)

            elif event_type == 'error':
                result_placeholder.error(f"Error: {content}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data via FastF1 & OpenF1 · Model: Llama 3.3 70B on Groq · Built by you")
