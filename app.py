import streamlit as st
from agent import run_strategy_agent

st.set_page_config(
    page_title="F1 Strategy Agent",
    page_icon="🏎️",
    layout="centered"
)

# ── API key ────────────────────────────────────────────────────────────────────
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = st.sidebar.text_input(
        "Groq API key", type="password",
        help="Get a free key at console.groq.com"
    )

# ── Session state init ─────────────────────────────────────────────────────────
# This is the memory layer — persists across Streamlit reruns
if "history" not in st.session_state:
    st.session_state.history = []       # full message history for the LLM
if "display" not in st.session_state:
    st.session_state.display = []       # (role, text) pairs for the UI

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏎️ F1 Strategy Agent")
st.caption("Powered by Llama 3.3 via Groq + FastF1 data")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Preset scenarios")
st.sidebar.caption("Click one to load it into the input box.")

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

st.sidebar.divider()
if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.history = []
    st.session_state.display = []
    st.rerun()

# ── Conversation history display ───────────────────────────────────────────────
for role, text in st.session_state.display:
    with st.chat_message(role):
        st.markdown(text)

# ── Input ──────────────────────────────────────────────────────────────────────
question = st.chat_input(
    "Ask a strategy question… e.g. 'What about Leclerc now?'"
    if st.session_state.display
    else "Describe the race situation…"
)

# Also allow loading a preset into the chat input area
if selected and not question:
    question = selected

# ── Run ────────────────────────────────────────────────────────────────────────
if question:
    if not api_key:
        st.warning("Add your Groq API key in the sidebar first.")
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.display.append(("user", question))

    # Run agent and stream output
    with st.chat_message("assistant"):
        with st.expander("Agent reasoning", expanded=False):
            tool_log = st.empty()
            log_lines = []

        answer_box = st.empty()

        with st.spinner("Thinking..."):
            for event_type, content in run_strategy_agent(
                question, api_key, st.session_state.history
            ):
                if event_type == 'tool_call':
                    log_lines.append(f"**Calling:** `{content}`")
                    tool_log.markdown("\n\n".join(log_lines))

                elif event_type == 'tool_result':
                    log_lines.append(f"```\n{content[:300]}\n```")
                    tool_log.markdown("\n\n".join(log_lines))

                elif event_type == 'answer':
                    answer_box.markdown(content)
                    st.session_state.display.append(("assistant", content))

                elif event_type == 'history':
                    # Save the updated full history for the next turn
                    st.session_state.history = content

                elif event_type == 'error':
                    answer_box.error(f"Error: {content}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data via FastF1 & OpenF1 · Model: Llama 3.3 70B on Groq · Built by you")
