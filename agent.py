import json
import requests
import os
import fastf1
import pandas as pd
from groq import Groq

os.makedirs('./f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('./f1_cache')

SYSTEM_PROMPT = """You are a Formula 1 strategy engineer working on the pit wall.
Your job is to analyze race situations and give clear, decisive pit stop recommendations.

When asked a strategy question:
1. Use the available tools to gather relevant data (tires, gaps, weather)
2. Reason through the undercut/overcut options
3. Give a clear recommendation with your reasoning

Always be specific: name the driver, the lap, the compound, and the expected outcome.
Think like an actual F1 engineer — decisive, data-driven, concise.

You have memory of the full conversation so far. If the user refers to a driver,
race, or situation mentioned earlier, use that context."""


# ── Compound colors (official F1 palette) ─────────────────────────────────────
COMPOUND_COLORS = {
    "SOFT":        "#E8002D",
    "MEDIUM":      "#FFF200",
    "HARD":        "#FFFFFF",
    "INTERMEDIATE":"#39B54A",
    "WET":         "#0067FF",
    "UNKNOWN":     "#888888",
}


# ── Tool functions ─────────────────────────────────────────────────────────────

def _load_driver_laps(year: int, race_name: str, driver_code: str) -> pd.DataFrame:
    """Shared helper — loads and returns cleaned lap DataFrame for a driver."""
    session = fastf1.get_session(year, race_name, 'R')
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps.pick_driver(driver_code)[
        ['LapNumber', 'Compound', 'LapTime', 'TyreLife']
    ].dropna(subset=['LapTime']).copy()
    laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
    laps['Driver'] = driver_code
    return laps


def get_tire_data(year: int, race_name: str, driver_code: str) -> str:
    """Returns lap-by-lap tire compound and lap time for a driver.
    Used to assess current tire age and degradation rate."""
    laps = _load_driver_laps(year, race_name, driver_code)
    lines = []
    for _, row in laps.iterrows():
        lines.append(
            f"Lap {int(row['LapNumber'])}: {row['Compound']} "
            f"(age {int(row['TyreLife'])} laps) — {row['LapTimeSec']:.3f}s"
        )
    return f"Tire data for {driver_code}:\n" + "\n".join(lines[-20:])


def get_chart_data(year: int, race_name: str, driver_code: str) -> dict:
    """
    Returns chart-ready lap data for a driver.
    Called alongside get_tire_data to generate the Plotly chart.
    """
    laps = _load_driver_laps(year, race_name, driver_code)
    return {
        "driver": driver_code,
        "race": race_name,
        "year": year,
        "laps": laps[['LapNumber', 'LapTimeSec', 'Compound', 'TyreLife']].to_dict('records')
    }


def get_race_gaps(year: int, race_name: str, driver_code: str) -> str:
    """Returns race position and cars ahead/behind a driver."""
    session = fastf1.get_session(year, race_name, 'R')
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps
    last_laps = laps.sort_values('LapNumber').groupby('Driver').last().reset_index()
    last_laps = last_laps[['Driver', 'Position', 'LapNumber']].dropna()
    last_laps['Position'] = last_laps['Position'].astype(int)
    last_laps = last_laps.sort_values('Position')
    try:
        driver_pos = last_laps[last_laps['Driver'] == driver_code]['Position'].values[0]
        car_ahead = last_laps[last_laps['Position'] == driver_pos - 1]['Driver'].values
        car_behind = last_laps[last_laps['Position'] == driver_pos + 1]['Driver'].values
        result = f"{driver_code} is in P{driver_pos}.\n"
        result += f"Car ahead: {car_ahead[0] if len(car_ahead) else 'leader'}\n"
        result += f"Car behind: {car_behind[0] if len(car_behind) else 'no one'}"
        return result
    except Exception as e:
        return f"Could not compute gaps: {str(e)}"


def compute_pit_window(
    current_lap: int,
    total_laps: int,
    gap_to_car_ahead_sec: float,
    avg_pit_loss_sec: float = 22.0
) -> str:
    """Computes whether undercut or overcut is viable."""
    laps_remaining = total_laps - current_lap
    undercut_viable = gap_to_car_ahead_sec < avg_pit_loss_sec
    overcut_viable = gap_to_car_ahead_sec > avg_pit_loss_sec and laps_remaining > 10
    result = f"Pit window analysis (lap {current_lap}/{total_laps}):\n"
    result += f"Gap to car ahead: {gap_to_car_ahead_sec:.1f}s | Pit loss: {avg_pit_loss_sec:.1f}s\n"
    result += f"Laps remaining: {laps_remaining}\n"
    result += f"Undercut viable: {'YES — pit now to jump them' if undercut_viable else 'NO — gap too large'}\n"
    result += f"Overcut viable: {'YES — stay out if tires hold' if overcut_viable else 'NO'}"
    return result


def get_circuit_weather(circuit_lat: float, circuit_lon: float) -> str:
    """Fetches current weather at a circuit using Open-Meteo. No API key needed."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={circuit_lat}&longitude={circuit_lon}"
        f"&current=temperature_2m,precipitation,weathercode,windspeed_10m"
        f"&hourly=precipitation_probability&forecast_days=1"
    )
    data = requests.get(url, timeout=10).json()
    current = data['current']
    rain_prob = data['hourly']['precipitation_probability'][0]
    weather_codes = {
        0: "clear", 1: "mainly clear", 2: "partly cloudy",
        3: "overcast", 51: "light drizzle", 61: "light rain",
        63: "moderate rain", 80: "rain showers"
    }
    condition = weather_codes.get(current['weathercode'], f"code {current['weathercode']}")
    return (
        f"Current conditions: {condition}\n"
        f"Temperature: {current['temperature_2m']}°C\n"
        f"Wind: {current['windspeed_10m']} km/h\n"
        f"Precipitation: {current['precipitation']} mm\n"
        f"Rain probability next hour: {rain_prob}%"
    )


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "get_tire_data": get_tire_data,
    "get_race_gaps": get_race_gaps,
    "compute_pit_window": compute_pit_window,
    "get_circuit_weather": get_circuit_weather,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_tire_data",
            "description": "Returns lap-by-lap tire compound and lap times for a driver.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "race_name": {"type": "string"},
                    "driver_code": {"type": "string"}
                },
                "required": ["year", "race_name", "driver_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_race_gaps",
            "description": "Returns race position and gap to cars ahead and behind.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "race_name": {"type": "string"},
                    "driver_code": {"type": "string"}
                },
                "required": ["year", "race_name", "driver_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_pit_window",
            "description": "Calculates whether undercut or overcut is viable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_lap": {"type": "integer"},
                    "total_laps": {"type": "integer"},
                    "gap_to_car_ahead_sec": {"type": "number"},
                    "avg_pit_loss_sec": {"type": "number"}
                },
                "required": ["current_lap", "total_laps", "gap_to_car_ahead_sec"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_circuit_weather",
            "description": "Fetches current weather at a circuit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "circuit_lat": {"type": "number"},
                    "circuit_lon": {"type": "number"}
                },
                "required": ["circuit_lat", "circuit_lon"]
            }
        }
    }
]


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_strategy_agent(question: str, api_key: str, history: list = None):
    """
    Runs the agent and yields (type, content) tuples.
    Types: 'tool_call', 'tool_result', 'chart', 'answer', 'history', 'error'
    """
    client = Groq(api_key=api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    for _ in range(6):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        except Exception as e:
            yield ('error', str(e))
            return

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            yield ('answer', msg.content)
            yield ('history', messages[1:])
            return

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            yield ('tool_call', f"{fn_name}({fn_args})")

            try:
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
            except Exception as e:
                result = f"Tool error: {str(e)}"

            # Whenever tire data is fetched, also generate chart data
            if fn_name == "get_tire_data" and "Tool error" not in str(result):
                try:
                    chart_data = get_chart_data(**fn_args)
                    yield ('chart', chart_data)
                except Exception:
                    pass  # chart failing should never break the agent

            yield ('tool_result', str(result)[:300])

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

    yield ('error', "Agent reached max iterations without a final answer.")
