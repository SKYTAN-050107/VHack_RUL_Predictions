import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from utils.auth import require_login, render_sidebar_user
from utils.mock_api_client import MockMLBackendClient
from utils.backend_api_client import BackendAPIClient
from utils.gemini_client import GeminiClient
from utils.ui_helpers import render_page_header, status_badge, render_status_badge
from config.settings import STATUS_COLORS, USE_MOCK_DATA, BACKEND_BASE_URL

st.set_page_config(page_title="Machine Analysis | DMC", page_icon="🔬", layout="wide")
require_login()

# ── Sidebar ───────────────────────────────────────────────────────────────────
client = MockMLBackendClient()
machines = client.fetch_all_machines()
machine_ids = [m.machine_id for m in machines]
machine_labels = {m.machine_id: f"{m.machine_id} — {m.name}" for m in machines}

with st.sidebar:
    st.markdown("## 🔬 Machine Analysis")
    st.divider()

    default_idx = 0
    pre_selected = st.session_state.get("selected_machine")
    if pre_selected and pre_selected in machine_ids:
        default_idx = machine_ids.index(pre_selected)

    selected_id = st.selectbox(
        "Select Machine",
        options=machine_ids,
        format_func=lambda x: machine_labels[x],
        index=default_idx,
    )
    st.session_state.selected_machine = selected_id

render_sidebar_user()

# ── Load data ─────────────────────────────────────────────────────────────────
machine  = client.fetch_machine_info(selected_id)
pred     = client.fetch_ml_prediction(selected_id)

render_page_header("🔬 Machine Analysis", f"{machine.name} | {machine.machine_id} | {machine.location}")

# ── Top Metrics Strip ─────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_status_badge(machine.status)
    st.caption("Status")
with c2:
    st.metric("Health Score", f"{pred.health_score:.1f}%")
with c3:
    color = STATUS_COLORS[machine.status]
    st.markdown(
        f"<div style='font-size:1.6em;font-weight:700;color:{color}'>{pred.rul_days:.0f} days</div>"
        f"<div style='color:#888;font-size:0.8em'>Remaining Useful Life</div>",
        unsafe_allow_html=True,
    )
with c4:
    st.metric("Vibration RMS", f"{pred.vibration_rms:.2f} mm/s",
              delta=f"{'⚠️ Above normal' if pred.vibration_rms > 1.5 else '✅ Normal'}", delta_color="off")
with c5:
    st.metric("Temperature", f"{pred.temperature_celsius:.1f}°C",
              delta=f"{'⚠️ Elevated' if pred.temperature_celsius > 60 else '✅ Normal'}", delta_color="off")

st.divider()

# ── Sensor Trends ─────────────────────────────────────────────────────────────
st.subheader("📈 Sensor Trends (Last 30 Days)")
tab_vib, tab_temp, tab_freq = st.tabs(["Vibration", "Temperature", "Frequency Spectrum"])
vib_df = client.fetch_machine_history(selected_id, "vibration", 30)
temp_df = client.fetch_machine_history(selected_id, "temperature", 30)


def _sensor_chart(df, y_col, y_label, normal_max, warning_max, unit):
    """Shared chart builder for vibration and temperature."""
    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[y_col],
        name=y_label, mode="lines",
        line=dict(color="#3498DB", width=1.5),
        hovertemplate=f"%{{y:.2f}} {unit}<extra></extra>",
    ))

    # Anomaly markers (points above warning threshold)
    anomalies = df[df[y_col] > warning_max]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies[y_col],
            name="Anomaly", mode="markers",
            marker=dict(color="#E74C3C", size=7, symbol="x"),
            hovertemplate=f"⚠️ ANOMALY: %{{y:.2f}} {unit}<extra></extra>",
        ))

    # Threshold bands
    fig.add_hline(y=normal_max, line_dash="dot",  line_color="#2ECC71", annotation_text=f"Normal max ({normal_max} {unit})")
    fig.add_hline(y=warning_max, line_dash="dash", line_color="#E74C3C", annotation_text=f"Critical ({warning_max} {unit})")

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1A1F2E",
        paper_bgcolor="#1A1F2E",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#2A2F3E"),
        yaxis=dict(title=f"{y_label} ({unit})", gridcolor="#2A2F3E"),
        showlegend=True,
        legend=dict(orientation="h", y=1.05),
    )
    return fig


with tab_vib:
    st.plotly_chart(_sensor_chart(vib_df, "vibration_rms", "Vibration RMS", 1.5, 3.0, "mm/s"), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Current", f"{pred.vibration_rms:.2f} mm/s")
    col2.metric("30-day Max", f"{vib_df['vibration_rms'].max():.2f} mm/s")
    col3.metric("30-day Avg", f"{vib_df['vibration_rms'].mean():.2f} mm/s")

with tab_temp:
    st.plotly_chart(_sensor_chart(temp_df, "temperature_celsius", "Temperature", 60.0, 75.0, "°C"), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Current", f"{pred.temperature_celsius:.1f}°C")
    col2.metric("30-day Max", f"{temp_df['temperature_celsius'].max():.1f}°C")
    col3.metric("30-day Avg", f"{temp_df['temperature_celsius'].mean():.1f}°C")

with tab_freq:
    fft_df = client.fetch_fft_spectrum(selected_id)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(
        x=fft_df["frequency_hz"], y=fft_df["amplitude"],
        fill="tozeroy", mode="lines",
        line=dict(color="#9B59B6", width=1),
        name="Amplitude",
        hovertemplate="f: %{x:.1f} Hz<br>Amp: %{y:.3f}<extra></extra>",
    ))
    # Mark significant peaks from prediction
    for peak in pred.frequency_peaks:
        fig_fft.add_vline(x=peak.frequency_hz, line_dash="dash", line_color="#E74C3C", opacity=0.7)
        fig_fft.add_annotation(
            x=peak.frequency_hz, y=peak.amplitude,
            text=peak.label, showarrow=True, arrowhead=2,
            font=dict(size=9, color="#E74C3C"), ay=-30,
        )
    fig_fft.update_layout(
        template="plotly_dark", plot_bgcolor="#1A1F2E", paper_bgcolor="#1A1F2E",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Frequency (Hz)", gridcolor="#2A2F3E"),
        yaxis=dict(title="Amplitude (mm/s)", gridcolor="#2A2F3E"),
    )
    st.plotly_chart(fig_fft, use_container_width=True)
    if pred.frequency_peaks:
        st.caption("**Detected frequency peaks:**")
        peaks_df = pd.DataFrame([fp.model_dump() for fp in pred.frequency_peaks])
        peaks_df.columns = ["Frequency (Hz)", "Amplitude", "Label"]
        st.dataframe(peaks_df, use_container_width=True, hide_index=True)


def _build_structured_facts(prediction, machine_status: str, vibration_df: pd.DataFrame, temperature_df: pd.DataFrame):
    vib_first = float(vibration_df["vibration_rms"].iloc[0])
    vib_last = float(vibration_df["vibration_rms"].iloc[-1])
    temp_first = float(temperature_df["temperature_celsius"].iloc[0])
    temp_last = float(temperature_df["temperature_celsius"].iloc[-1])

    vib_delta = vib_last - vib_first
    temp_delta = temp_last - temp_first
    rul_drop_estimate = max(0.0, prediction.rul_days * (prediction.anomaly_score * 0.25))

    top_drivers = [
        {
            "feature": "vibration_rms",
            "rank": 1,
            "occurrence_pct": round(min(100.0, max(0.0, 55.0 + (vib_delta * 10.0))), 2),
            "avg_shap_value": round(vib_delta, 4),
            "direction": "increase_risk" if vib_delta >= 0 else "decrease_risk",
        },
        {
            "feature": "temperature_celsius",
            "rank": 2,
            "occurrence_pct": round(min(100.0, max(0.0, 45.0 + (temp_delta * 4.0))), 2),
            "avg_shap_value": round(temp_delta, 4),
            "direction": "increase_risk" if temp_delta >= 0 else "decrease_risk",
        },
    ]

    anomalies = []
    if vib_last > 3.0:
        anomalies.append("Vibration is above critical threshold.")
    elif vib_last > 1.5:
        anomalies.append("Vibration is above warning threshold.")
    if temp_last > 75.0:
        anomalies.append("Temperature is above critical threshold.")
    elif temp_last > 60.0:
        anomalies.append("Temperature is above warning threshold.")

    risk_band = "Green"
    if machine_status == "Critical" or prediction.rul_days <= 20:
        risk_band = "Red"
    elif machine_status == "Warning" or prediction.rul_days <= 60:
        risk_band = "Yellow"

    return {
        "period": {
            "days": 30,
            "start": str(vibration_df["timestamp"].iloc[0]),
            "end": str(vibration_df["timestamp"].iloc[-1]),
        },
        "rul": {
            "rul_now": round(float(prediction.rul_days), 2),
            "rul_delta": round(-rul_drop_estimate, 2),
            "rul_delta_pct": round((-rul_drop_estimate / max(float(prediction.rul_days), 1.0)) * 100.0, 2),
            "hourly_slope": round((-rul_drop_estimate / (30.0 * 24.0)), 4),
        },
        "risk_band": risk_band,
        "confidence": "high" if prediction.confidence >= 0.8 else "medium" if prediction.confidence >= 0.6 else "low",
        "top_drivers": top_drivers,
        "anomalies": anomalies,
    }


st.subheader("📉 Why RUL Is Dropping")
facts_key = f"rul_facts_{selected_id}"
explain_key = f"rul_explain_{selected_id}"
api_status_key = f"rul_backend_status_{selected_id}"


def _machine_id_for_backend(machine_id: str):
    if isinstance(machine_id, int):
        return machine_id
    digits = "".join(ch for ch in str(machine_id) if ch.isdigit())
    if digits:
        return int(digits)
    return None

if st.button("Generate Driver Trend Explanation", icon="🧭"):
    facts_payload = {}
    explanation_payload = {}
    st.session_state[api_status_key] = "fallback"

    backend_machine_id = _machine_id_for_backend(selected_id)
    if not USE_MOCK_DATA and backend_machine_id is not None:
        backend_client = BackendAPIClient(BACKEND_BASE_URL)
        api_payload = backend_client.fetch_driver_trend(
            machine_id=backend_machine_id,
            hours_lookback=24,
            top_n=5,
            dataset_id="FD001",
        )
        if api_payload.get("status") == "ok":
            facts_payload = api_payload.get("structured_facts", {})
            explanation_payload = api_payload.get("llm_explanation", {})
            st.session_state[api_status_key] = "backend"

    if not facts_payload or not explanation_payload:
        facts_payload = _build_structured_facts(pred, machine.status, vib_df, temp_df)
        gemini = GeminiClient()
        explanation_payload = gemini.generate_rul_drop_explanation_from_facts(
            machine_id=selected_id,
            machine=machine,
            prediction=pred,
            structured_facts=facts_payload,
        )

    st.session_state[facts_key] = facts_payload
    st.session_state[explain_key] = explanation_payload

if explain_key in st.session_state:
    explanation_payload = st.session_state[explain_key]
    facts_payload = st.session_state.get(facts_key, {})

    if st.session_state.get(api_status_key) == "backend":
        st.caption("Source: backend /api/machines/{machine_id}/driver-trend")
    else:
        st.caption("Source: local structured-facts fallback")

    summary_text = explanation_payload.get("summary") or explanation_payload.get("explanation") or "No explanation available."
    st.markdown(summary_text)
    st.caption(f"Confidence: {explanation_payload.get('confidence', 'unknown')}")

    action_cols = st.columns(4)
    actions = (explanation_payload.get("actions") or [{}])[0]
    action_cols[0].metric("Inspect First", actions.get("inspect_first", "N/A"))
    action_cols[1].metric("Check Now", actions.get("check_now", "N/A"))
    action_cols[2].metric("Plan By", actions.get("plan_by", "N/A"))
    action_cols[3].metric("Escalate If", actions.get("escalate_if", "N/A"))

    with st.expander("Structured Facts Used"):
        st.json(facts_payload)
else:
    st.info("Generate explanation to produce an operator-friendly summary from structured facts only.")

st.divider()

# ── Root Cause Analysis ───────────────────────────────────────────────────────
st.subheader("🧠 AI Root Cause Analysis")

cache_key = f"rca_{selected_id}"
col_btn, col_conf = st.columns([1, 3])
with col_btn:
    run_analysis = st.button("Generate Analysis", type="primary", icon="🤖")
with col_conf:
    st.caption(f"ML Confidence: **{pred.confidence:.0%}** | Anomaly Score: **{pred.anomaly_score:.3f}**")

if run_analysis or cache_key in st.session_state.get("generated_analysis", {}):
    if run_analysis:
        with st.spinner("Analysing sensor data with AI..."):
            gemini = GeminiClient()
            result = gemini.generate_root_cause(selected_id, pred, machine)
            st.session_state.generated_analysis[cache_key] = result

    analysis_text = st.session_state.generated_analysis.get(cache_key, "")

    # Parse out sections for tabbed display
    tab_detail, tab_exec, tab_tech = st.tabs(["📋 Detailed Technical", "📊 Executive Summary", "🔧 Technician Checklist"])

    with tab_detail:
        st.markdown(analysis_text)

    with tab_exec:
        # Extract Executive Summary section
        exec_start = analysis_text.find("### Executive Summary")
        exec_end   = analysis_text.find("###", exec_start + 1) if exec_start != -1 else -1
        if exec_start != -1:
            exec_text = analysis_text[exec_start:exec_end if exec_end != -1 else None]
            st.markdown(exec_text)
        else:
            st.markdown(analysis_text)

    with tab_tech:
        # Extract Technician Guidance section
        tech_start = analysis_text.find("### Technician Guidance")
        tech_end   = analysis_text.find("###", tech_start + 1) if tech_start != -1 else -1
        if tech_start != -1:
            tech_text = analysis_text[tech_start:tech_end if tech_end != -1 else None]
            st.markdown(tech_text)
        else:
            st.markdown(analysis_text)
else:
    st.info("Click **Generate Analysis** to run AI-powered root cause analysis on this machine's sensor data.")

    if pred.anomalies:
        st.subheader("🚩 Detected Anomalies")
        for anomaly in pred.anomalies:
            st.markdown(f"- {anomaly}")

st.divider()

# ── Degradation Timeline ──────────────────────────────────────────────────────
st.subheader("📅 Degradation Timeline")

# Build synthetic event markers based on prediction data
events = []
now = datetime.now()

if machine.status in ["Warning", "Critical"]:
    events.append({"date": now - pd.Timedelta(days=45), "event": "Baseline vibration increase first detected (+8%)", "severity": "Low"})
    events.append({"date": now - pd.Timedelta(days=28), "event": "CUSUM algorithm flagged statistically significant trend change", "severity": "Medium"})
    events.append({"date": now - pd.Timedelta(days=14), "event": "Temperature rise becomes significant (+4°C above 30-day avg)", "severity": "Medium"})

if machine.status == "Critical":
    events.append({"date": now - pd.Timedelta(days=7),  "event": "Vibration entered WARNING zone (>1.5 mm/s)", "severity": "High"})
    events.append({"date": now - pd.Timedelta(days=3),  "event": "RUL prediction dropped below 20-day critical threshold", "severity": "Critical"})
    events.append({"date": now - pd.Timedelta(days=1),  "event": "Anomaly score exceeded 0.8 — rapid degradation phase entered", "severity": "Critical"})

events.append({"date": now, "event": "Current state — maintenance decision required", "severity": machine.status})

if events:
    severity_colors = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E67E22", "Critical": "#E74C3C", "Healthy": "#2ECC71", "Warning": "#F39C12"}
    fig_tl = go.Figure()
    for i, ev in enumerate(events):
        color = severity_colors.get(ev["severity"], "#888")
        fig_tl.add_trace(go.Scatter(
            x=[ev["date"]], y=[0],
            mode="markers+text",
            marker=dict(size=16, color=color, symbol="circle"),
            text=[ev["event"]],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=f"<b>{ev['date'].strftime('%Y-%m-%d')}</b><br>{ev['event']}<extra></extra>",
            showlegend=False,
        ))

    fig_tl.update_layout(
        template="plotly_dark", plot_bgcolor="#1A1F2E", paper_bgcolor="#1A1F2E",
        height=200, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="#2A2F3E"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=True, zerolinecolor="#2A2F3E"),
    )
    st.plotly_chart(fig_tl, use_container_width=True)
