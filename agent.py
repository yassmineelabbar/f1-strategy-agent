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
race, or situation mentioned earlier, use that context.

Tool routing rules:
- If the user mentions 'latest', 'current', 'most recent', or 'now' → use get_latest_session first.
- If the user specifies a year and race name → use get_openf1_session.
- Never assume the current year is 2024. The actual current year is 2026."""


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

# ── OpenF1 tools (live + recent sessions) ─────────────────────────────────────

OPENF1_BASE = "https://api.openf1.org/v1"


def get_openf1_session(year: int, race_name: str) -> str:
    """
    Finds an F1 session by year and race name using OpenF1.
    Returns the session key needed for other OpenF1 tools.
    Use this first before calling get_live_standings or get_live_stints.
    """
    resp = requests.get(
        f"{OPENF1_BASE}/sessions",
        params={"year": year, "session_name": "Race"},
        timeout=10
    ).json()

    if not resp:
        return f"No sessions found for year {year}."

    # Match race name loosely
    race_lower = race_name.lower()
    match = None
    for s in resp:
        location = s.get("location", "").lower()
        country = s.get("country_name", "").lower()
        circuit = s.get("circuit_short_name", "").lower()
        if any(race_lower in field for field in [location, country, circuit]):
            match = s
            break

    if not match:
        # Fall back to latest session
        match = resp[-1]

    return (
        f"Session found: {match.get('location')} {match.get('year')} — "
        f"{match.get('session_name')}\n"
        f"Session key: {match.get('session_key')}\n"
        f"Date: {match.get('date_start', 'unknown')}\n"
        f"Circuit: {match.get('circuit_short_name')}"
    )


def get_live_standings(session_key: int) -> str:
    """
    Returns current race standings for all drivers in a session.
    Shows position, driver code, team, gap to leader, and current lap.
    Use after get_openf1_session to get the session_key.
    """
    # Get drivers in this session
    drivers_resp = requests.get(
        f"{OPENF1_BASE}/drivers",
        params={"session_key": session_key},
        timeout=10
    ).json()

    if not drivers_resp:
        return f"No driver data found for session {session_key}."

    # Map driver_number → driver info
    driver_map = {
        d["driver_number"]: {
            "code": d.get("name_acronym", "???"),
            "team": d.get("team_name", "?"),
        }
        for d in drivers_resp
    }

    # Get latest position for each driver
    positions_resp = requests.get(
        f"{OPENF1_BASE}/position",
        params={"session_key": session_key},
        timeout=10
    ).json()

    if not positions_resp:
        return f"No position data found for session {session_key}."

    # Keep only the latest position entry per driver
    latest = {}
    for p in positions_resp:
        dn = p["driver_number"]
        if dn not in latest or p["date"] > latest[dn]["date"]:
            latest[dn] = p

    # Sort by position
    sorted_pos = sorted(latest.values(), key=lambda x: x.get("position", 99))

    lines = [f"Standings for session {session_key}:"]
    for p in sorted_pos[:10]:  # top 10
        dn = p["driver_number"]
        info = driver_map.get(dn, {"code": str(dn), "team": "?"})
        lines.append(
            f"P{p['position']} — {info['code']} ({info['team']})"
        )

    return "\n".join(lines)


def get_live_stints(session_key: int, driver_number: int) -> str:
    """
    Returns tire stint history for a driver in a session.
    Shows compound, start lap, end lap, and stint duration.
    Use after get_openf1_session to get the session_key.
    """
    resp = requests.get(
        f"{OPENF1_BASE}/stints",
        params={"session_key": session_key, "driver_number": driver_number},
        timeout=10
    ).json()

    if not resp:
        return f"No stint data for driver {driver_number} in session {session_key}."

    # Also get driver name
    drivers = requests.get(
        f"{OPENF1_BASE}/drivers",
        params={"session_key": session_key, "driver_number": driver_number},
        timeout=10
    ).json()
    name = drivers[0].get("name_acronym", str(driver_number)) if drivers else str(driver_number)

    lines = [f"Tire stints for {name} (session {session_key}):"]
    for s in resp:
        lap_start = s.get("lap_start", "?")
        lap_end = s.get("lap_end", "current")
        compound = s.get("compound", "?")
        age = s.get("tyre_age_at_start", 0)
        lines.append(
            f"Stint {s.get('stint_number', '?')}: {compound} — "
            f"laps {lap_start} to {lap_end} (age at start: {age})"
        )

    return "\n".join(lines)
def get_latest_race_info() -> str:
    """
    Returns the most recently completed F1 race AND its standings in one call.
    Use this whenever the user asks about the current race, latest session,
    most recent race, or current standings without specifying a year or race name.
    """
    from datetime import datetime, timezone

    resp = requests.get(
        f"{OPENF1_BASE}/sessions",
        params={"session_name": "Race"},
        timeout=10
    ).json()

    if not resp or not isinstance(resp, list):
        return f"OpenF1 API returned unexpected data: {str(resp)[:200]}"

    # Filter to sessions that have already started (date_start <= now)
    now = datetime.now(timezone.utc)
    past_sessions = []
    for s in resp:
        if not isinstance(s, dict):
            continue
        date_str = s.get("date_start", "")
        try:
            session_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if session_dt <= now:
                past_sessions.append(s)
        except Exception:
            continue

    if not past_sessions:
        return "No completed sessions found."

    # Most recent completed session
    latest = past_sessions[-1]
    session_key = latest.get('session_key')

    result = (
        f"Most recent completed session: {latest.get('location')} {latest.get('year')} — "
        f"{latest.get('session_name')}\n"
        f"Session key: {session_key}\n"
        f"Date: {latest.get('date_start', 'unknown')}\n"
        f"Circuit: {latest.get('circuit_short_name')}\n\n"
    )

    if not session_key:
        return result + "Could not fetch standings — no session key."

    # Fetch drivers
    drivers_resp = requests.get(
        f"{OPENF1_BASE}/drivers",
        params={"session_key": session_key},
        timeout=10
    ).json()

    driver_map = {}
    if isinstance(drivers_resp, list):
        driver_map = {
            d["driver_number"]: {
                "code": d.get("name_acronym", "???"),
                "team": d.get("team_name", "?"),
            }
            for d in drivers_resp if isinstance(d, dict)
        }

    # Fetch positions
    positions_resp = requests.get(
        f"{OPENF1_BASE}/position",
        params={"session_key": session_key},
        timeout=10
    ).json()

    if isinstance(positions_resp, list) and positions_resp:
        latest_pos = {}
        for p in positions_resp:
            if not isinstance(p, dict):
                continue
            dn = p.get("driver_number")
            if dn and (dn not in latest_pos or p.get("date", "") > latest_pos[dn].get("date", "")):
                latest_pos[dn] = p

        sorted_pos = sorted(latest_pos.values(), key=lambda x: x.get("position", 99))
        result += "Final standings:\n"
        for p in sorted_pos[:10]:
            dn = p["driver_number"]
            info = driver_map.get(dn, {"code": str(dn), "team": "?"})
            result += f"P{p['position']} — {info['code']} ({info['team']})\n"
    else:
        result += f"Standings unavailable: {str(positions_resp)[:150]}"

    return result

# ── Tool registry ──────────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "get_tire_data": get_tire_data,
    "get_race_gaps": get_race_gaps,
    "compute_pit_window": compute_pit_window,
    "get_circuit_weather": get_circuit_weather,
    "get_openf1_session": get_openf1_session,
    "get_live_standings": get_live_standings,
    "get_live_stints": get_live_stints,
    "get_latest_race_info": get_latest_race_info,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_tire_data",
            "description": "Returns lap-by-lap tire compound and lap times for a driver using FastF1. Best for detailed historical analysis.",
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
            "description": "Returns race position and gap to cars ahead and behind using FastF1. Best for historical races.",
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
            "description": "Calculates whether undercut or overcut is viable given lap number, total laps, and gap to car ahead.",
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
            "description": "Fetches current weather at a circuit. Use for wet or intermediate tire decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "circuit_lat": {"type": "number"},
                    "circuit_lon": {"type": "number"}
                },
                "required": ["circuit_lat", "circuit_lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_openf1_session",
            "description": "Finds an F1 session by year and race name using OpenF1. Returns the session key needed for get_live_standings and get_live_stints. Use this first for any live or recent session query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Season year e.g. 2024"},
                    "race_name": {"type": "string", "description": "Race location e.g. 'Monaco', 'Silverstone'"}
                },
                "required": ["year", "race_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_standings",
            "description": "Returns current race standings — positions, drivers, teams — from OpenF1. Use after get_openf1_session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer", "description": "Session key from get_openf1_session"}
                },
                "required": ["session_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_stints",
            "description": "Returns tire stint history for a specific driver from OpenF1 — compound, start/end lap, tyre age. Use after get_openf1_session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer", "description": "Session key from get_openf1_session"},
                    "driver_number": {"type": "integer", "description": "Driver race number e.g. 1 for Verstappen, 4 for Norris"}
                },
                "required": ["session_key", "driver_number"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "get_latest_race_info",
        "description": "Returns the most recent F1 race session AND its current standings in a single call. Use this for ANY query about the current race, latest session, most recent race, or current standings. Do NOT use get_openf1_session or get_live_standings for 'latest' queries — always use this instead.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
           }
        }
    },
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
                temperature=0,
                max_tokens=1024,
                
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
            fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

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
