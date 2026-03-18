import streamlit as st
import plotly.graph_objects as go
from agent import run_strategy_agent, COMPOUND_COLORS

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

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "display" not in st.session_state:
    st.session_state.display = []  # list of dicts: {role, type, content}


# ── Chart builder ──────────────────────────────────────────────────────────────
def build_lap_chart(charts: list) -> go.Figure:
    """
    Takes a list of chart_data dicts (one per driver) and returns
    a Plotly figure with lap times colored by tire compound.
    """
    fig = go.Figure()

    for chart in charts:
        driver = chart['driver']
        laps = chart['laps']
        if not laps:
            continue

        # Group consecutive laps by compound to draw colored segments
        current_compound = None
        seg_laps, seg_times = [], []

        def flush_segment():
            if seg_laps and current_compound:
                color = COMPOUND_COLORS.get(current_compound.upper(), "#888888")
                fig.add_trace(go.Scatter(
                    x=seg_laps,
                    y=seg_times,
                    mode='lines+markers',
                    name=f"{driver} — {current_compound.capitalize()}",
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color,
                                line=dict(width=1, color='#333')),
                    legendgroup=driver,
                    hovertemplate=(
                        f"<b>{driver}</b><br>"
                        "Lap %{x}<br>"
                        "Time: %{y:.3f}s<br>"
                        f"Compound: {current_compound.capitalize()}"
                        "<extra></extra>"
                    )
                ))

        for lap in laps:
            compound = lap.get('Compound', 'UNKNOWN')
            if compound != current_compound:
                flush_segment()
                current_compound = compound
                seg_laps = [lap['LapNumber']]
                seg_times = [lap['LapTimeSec']]
            else:
                seg_laps.append(lap['LapNumber'])
                seg_times.append(lap['LapTimeSec'])
        flush_segment()

    race = charts[0]['race'] if charts else ""
    year = charts[0]['year'] if charts else ""

    fig.update_layout(
        title=dict(
            text=f"{race} {year} — Lap times by compound",
            font=dict(size=14)
        ),
        xaxis=dict(title="Lap", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Lap time (s)", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=60, b=0),
        hovermode="x unified",
        height=340,
    )
    return fig


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

# ── Render conversation history ────────────────────────────────────────────────
for item in st.session_state.display:
    with st.chat_message(item["role"]):
        if item["type"] == "text":
            st.markdown(item["content"])
        elif item["type"] == "chart":
            fig = build_lap_chart(item["content"])
            st.plotly_chart(fig, use_container_width=True)

# ── Input ──────────────────────────────────────────────────────────────────────
question = st.chat_input(
    "Ask a follow-up… e.g. 'What about Verstappen?'"
    if st.session_state.display else
    "Describe the race situation…"
)
if selected and not question:
    question = selected

# ── Run agent ──────────────────────────────────────────────────────────────────
if question:
    if not api_key:
        st.warning("Add your Groq API key in the sidebar first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.display.append({"role": "user", "type": "text", "content": question})

    with st.chat_message("assistant"):
        with st.expander("Agent reasoning", expanded=False):
            tool_log = st.empty()
            log_lines = []

        chart_placeholder = st.empty()
        answer_box = st.empty()

        pending_charts = []  # accumulate charts across tool calls this turn

        with st.spinner("Thinking..."):
            for event_type, content in run_strategy_agent(
                question, api_key, st.session_state.history
            ):
                if event_type == 'tool_call':
                    log_lines.append(f"**Calling:** `{content}`")
                    tool_log.markdown("\n\n".join(log_lines))

                elif event_type == 'tool_result':
                    log_lines.append(f"```\n{content}\n```")
                    tool_log.markdown("\n\n".join(log_lines))

                elif event_type == 'chart':
                    pending_charts.append(content)
                    # Show chart immediately as data arrives
                    fig = build_lap_chart(pending_charts)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                elif event_type == 'answer':
                    answer_box.markdown(content)
                    # Save chart and answer to display history
                    if pending_charts:
                        st.session_state.display.append({
                            "role": "assistant",
                            "type": "chart",
                            "content": pending_charts
                        })
                    st.session_state.display.append({
                        "role": "assistant",
                        "type": "text",
                        "content": content
                    })

                elif event_type == 'history':
                    st.session_state.history = content

                elif event_type == 'error':
                    answer_box.error(f"Error: {content}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data via FastF1 & OpenF1 · Model: Llama 3.3 70B on Groq · Built by yassmineelabbar")
```

---

## Updated `requirements.txt`
```
fastf1
groq
requests
streamlit
plotly
pandas
