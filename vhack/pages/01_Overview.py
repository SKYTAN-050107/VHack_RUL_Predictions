import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Overview", layout="wide")


@st.cache_data(ttl=2, show_spinner=False)
def get_range_info(dataset_id: str, split: str, unit_id: int):
    response = requests.get(
        f"{API_URL}/machines/replay/range-info",
        params={"dataset_id": dataset_id, "split": split, "unit_id": unit_id},
        timeout=12,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=2, show_spinner=False)
def get_replay_dashboard(
    dataset_id: str,
    split: str,
    unit_id: int,
    model_mode: str,
    start_cycle: int,
    end_cycle: int,
    shap_enabled: bool,
    shap_sample_interval: int,
):
    response = requests.get(
        f"{API_URL}/machines/dashboard/replay",
        params={
            "dataset_id": dataset_id,
            "split": split,
            "unit_id": unit_id,
            "model_mode": model_mode,
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "telemetry_limit": 60,
            "shap_enabled": bool(shap_enabled),
            "shap_sample_interval": int(shap_sample_interval),
        },
        timeout=18,
    )
    response.raise_for_status()
    return response.json()


st.title("C-MAPSS Replay Overview")

PAGE_KEY = "root_replay"
STEP_CYCLES = 4

with st.sidebar:
    st.subheader("Replay Controls")
    poll_seconds = st.slider("Polling interval (seconds)", min_value=1, max_value=10, value=2)
    live_polling = st.checkbox("Live polling", value=True)
    dataset_id = st.selectbox("Dataset", ["FD001", "FD002", "FD003", "FD004"], index=0)
    split = st.selectbox("Split", ["test", "train"], index=0)

    unit_hint = st.session_state.get("replay_unit_id", 1)
    try:
        initial_info = get_range_info(dataset_id=dataset_id, split=split, unit_id=unit_hint)
    except Exception:
        initial_info = {"available_units": [1], "model_modes": {"base": {"available": True, "message": "Unknown"}, "adapted": {"available": False, "message": "Unknown"}}}

    available_units = initial_info.get("available_units", [1])
    if not available_units:
        available_units = [1]

    default_idx = 0
    if unit_hint in available_units:
        default_idx = available_units.index(unit_hint)

    unit_id = st.selectbox("Engine Unit", options=available_units, index=default_idx)
    st.session_state["replay_unit_id"] = unit_id

    model_modes = initial_info.get("model_modes", {})
    adapted_available = bool(model_modes.get("adapted", {}).get("available", False))
    mode_options = ["base", "adapted"] if adapted_available else ["base"]
    model_mode = st.selectbox("Model Mode", mode_options, index=0)
    shap_enabled = st.checkbox("Show SHAP", value=False)
    shap_sample_interval = st.select_slider("SHAP sample interval", options=[5, 10, 20], value=10)

    if not adapted_available:
        st.caption(f"Adapted mode unavailable: {model_modes.get('adapted', {}).get('message', 'N/A')}")

try:
    range_info = get_range_info(dataset_id=dataset_id, split=split, unit_id=unit_id)
except Exception as exc:
    st.error(f"Failed to load replay range info: {str(exc)}")
    st.stop()

range_start = int(range_info.get("range_start", 1))
range_end = int(range_info.get("range_end", range_start))

context_key = f"{PAGE_KEY}_context"
playhead_key = f"{PAGE_KEY}_playhead"
playing_key = f"{PAGE_KEY}_is_playing"
speed_key = f"{PAGE_KEY}_speed"
loop_key = f"{PAGE_KEY}_loop"
refresh_key = f"{PAGE_KEY}_last_refresh_count"
tick_time_key = f"{PAGE_KEY}_last_tick_time"
frame_key = f"{PAGE_KEY}_last_replay"

context = (dataset_id, split, unit_id, model_mode, bool(shap_enabled), int(shap_sample_interval))
if st.session_state.get(context_key) != context:
    st.session_state[context_key] = context
    st.session_state[playhead_key] = range_start
    st.session_state[playing_key] = False
    st.session_state[speed_key] = st.session_state.get(speed_key, 1)
    st.session_state[loop_key] = st.session_state.get(loop_key, False)
    st.session_state[refresh_key] = None
    st.session_state[tick_time_key] = None
    st.session_state[frame_key] = None

playhead_cycle = int(st.session_state.get(playhead_key, range_start))
playhead_cycle = max(range_start, min(playhead_cycle, range_end))
st.session_state[playhead_key] = playhead_cycle

with st.sidebar:
    speed_multiplier = st.select_slider("Speed", options=[1, 2, 4], value=int(st.session_state.get(speed_key, 1)), format_func=lambda x: f"{x}x")
    st.session_state[speed_key] = speed_multiplier
    st.session_state[loop_key] = st.checkbox("Loop at end", value=bool(st.session_state.get(loop_key, False)))

    st.session_state[playing_key] = st.toggle("Play", value=bool(st.session_state.get(playing_key, False)))
    if st.button("Reset", use_container_width=True):
        st.session_state[playhead_key] = range_start
        st.session_state[playing_key] = False
        st.session_state[tick_time_key] = None
        st.rerun()

def _advance_playhead():
    if not st.session_state.get(playing_key, False):
        return
    step = STEP_CYCLES * int(st.session_state.get(speed_key, 1))
    next_cycle = int(st.session_state.get(playhead_key, range_start)) + step
    if next_cycle > range_end:
        if st.session_state.get(loop_key, False):
            next_cycle = range_start
        else:
            next_cycle = range_end
            st.session_state[playing_key] = False
    st.session_state[playhead_key] = next_cycle


def _render_frame(fetch: bool):
    active_playhead = int(st.session_state.get(playhead_key, range_start))
    replay = st.session_state.get(frame_key)
    if fetch or replay is None:
        try:
            replay = get_replay_dashboard(
                dataset_id=dataset_id,
                split=split,
                unit_id=unit_id,
                model_mode=model_mode,
                start_cycle=range_start,
                end_cycle=active_playhead,
                shap_enabled=shap_enabled,
                shap_sample_interval=shap_sample_interval,
            )
            st.session_state[frame_key] = replay
        except Exception as exc:
            st.error(f"Replay endpoint unavailable: {str(exc)}")
            return

    metadata = replay.get("metadata", {})
    series = replay.get("series", [])
    kpi = replay.get("kpi")
    shap_timeline = replay.get("shap_timeline", {})
    shap_meta = replay.get("shap_meta", {})

    frame_header = st.container()
    with frame_header:
        st.caption(
            f"Engine Unit {metadata.get('unit_id', unit_id)} "
            f"({metadata.get('dataset_id', dataset_id)} {metadata.get('split', split)}) | "
            f"Cycle {active_playhead}/{range_end} | "
            f"Mode: {model_mode.upper()}"
        )

    if not series:
        st.info("No replay points in selected range.")
        return

    df = pd.DataFrame(series).sort_values("cycle")

    kpi_box = st.container()
    with kpi_box:
        metric_cols = st.columns(4)
        if kpi:
            metric_cols[0].metric("Predicted RUL", f"{kpi.get('predicted_rul', 0):.2f}", delta=kpi.get("predicted_delta"))
            metric_cols[1].metric("Actual RUL", f"{kpi.get('actual_rul', 0):.2f}", delta=kpi.get("actual_delta"))
            metric_cols[2].metric("Abs Error", f"{kpi.get('abs_error', 0):.2f}")
        metric_cols[3].metric("Points", metadata.get("points_in_range", len(df)))

    charts_box = st.container()
    with charts_box:
        st.subheader("RUL Tracking")
        st.line_chart(df.set_index("cycle")[["actual_rul", "predicted_rul"]])

        if "abs_error" in df.columns:
            st.subheader("Absolute Error")
            st.line_chart(df.set_index("cycle")[["abs_error"]])

        st.subheader("Sensor Trends")
        sensor_cols = [
            c
            for c in df.columns
            if c in {"vibration", "temperature", "load"} or str(c).startswith("aux_")
        ]
        if sensor_cols:
            st.line_chart(df.set_index("cycle")[sensor_cols])
            st.caption(f"Showing {len(sensor_cols)} sensor features used for replay inference.")

    table_box = st.container()
    with table_box:
        st.dataframe(
            df[[c for c in ["cycle", "predicted_rul", "actual_rul", "abs_error", "health_state"] if c in df.columns]],
            hide_index=True,
            use_container_width=True,
        )

    if bool(shap_enabled):
        shap_box = st.container()
        with shap_box:
            st.subheader("Feature Importance (SHAP)")
            cycle_key = str(active_playhead)
            cycle_shap = shap_timeline.get(cycle_key)
            if not cycle_shap:
                st.info(f"SHAP not sampled at cycle {active_playhead}. Interval: {shap_meta.get('sample_interval', shap_sample_interval)}")
            else:
                top_features = cycle_shap.get("top_features", [])
                if top_features:
                    shap_df = pd.DataFrame(top_features)
                    if "feature" in shap_df.columns and "shap_value" in shap_df.columns:
                        st.bar_chart(shap_df.set_index("feature")[["shap_value"]])
                    st.dataframe(
                        shap_df[[c for c in ["rank", "feature", "shap_value", "direction"] if c in shap_df.columns]],
                        hide_index=True,
                        use_container_width=True,
                    )
                st.caption(
                    f"SHAP mode: {cycle_shap.get('mode', 'N/A')} | "
                    f"Cache hits: {shap_meta.get('cache_hits', 0)} | "
                    f"Cache misses: {shap_meta.get('cache_misses', 0)} | "
                    f"Compute: {shap_meta.get('compute_ms', 0)} ms"
                )
                if shap_meta.get("note"):
                    st.caption(str(shap_meta.get("note")))


if hasattr(st, "fragment"):
    @st.fragment(run_every=f"{poll_seconds}s" if live_polling else None)
    def _live_fragment():
        is_playing = bool(st.session_state.get(playing_key, False))
        if is_playing:
            _advance_playhead()
        _render_frame(fetch=is_playing)

    _live_fragment()
else:
    _render_frame(fetch=True)
