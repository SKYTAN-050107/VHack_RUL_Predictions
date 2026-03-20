import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils.auth import require_login, render_sidebar_user
from utils.mock_api_client import MockMLBackendClient
from utils.ui_helpers import render_page_header, status_badge, render_critical_alert
from config.settings import STATUS_COLORS, STATUS_ICONS

st.set_page_config(page_title="Dashboard | DMC", page_icon="📊", layout="wide")
require_login()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ DMC")
    st.caption("Digital Machinery Caretaker")
    st.divider()
    st.markdown("**Filter Fleet**")
    status_filter = st.multiselect(
        "Status",
        ["Healthy", "Warning", "Critical"],
        default=["Healthy", "Warning", "Critical"],
    )
    location_filter = st.text_input("Location contains", "")

render_sidebar_user()

# ── Data ──────────────────────────────────────────────────────────────────────
client = MockMLBackendClient()
machines = client.fetch_all_machines()
df = pd.DataFrame([m.model_dump() for m in machines])

# Apply sidebar filters
if status_filter:
    df = df[df["status"].isin(status_filter)]
if location_filter:
    df = df[df["location"].str.contains(location_filter, case=False, na=False)]

all_machines = client.fetch_all_machines()  # unfiltered, for KPI cards

# ── Page Header ───────────────────────────────────────────────────────────────
render_page_header("📊 Fleet Dashboard", f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {len(all_machines)} machines monitored")

# ── KPI Metrics ───────────────────────────────────────────────────────────────
all_df = pd.DataFrame([m.model_dump() for m in all_machines])
healthy_count  = int((all_df["status"] == "Healthy").sum())
warning_count  = int((all_df["status"] == "Warning").sum())
critical_count = int((all_df["status"] == "Critical").sum())
avg_rul        = float(all_df["rul_days"].mean())
avg_health     = float(all_df["health_score"].mean())

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("🟢 Healthy", healthy_count, help="Machines operating within normal parameters")
with c2:
    st.metric("🟡 Warning", warning_count, help="Machines requiring attention within 60 days")
with c3:
    st.metric("🔴 Critical", critical_count,
              delta=f"{critical_count} urgent" if critical_count > 0 else None,
              delta_color="inverse",
              help="Machines requiring urgent maintenance")
with c4:
    st.metric("⏱️ Avg RUL", f"{avg_rul:.0f} days", help="Average Remaining Useful Life across all machines")
with c5:
    st.metric("💚 Fleet Health", f"{avg_health:.1f}%", help="Average health score across all machines")

st.divider()

# ── Critical Alerts ──────────────────────────────────────────────────────────
critical_machines = all_df[all_df["status"] == "Critical"].sort_values("rul_days")
if not critical_machines.empty:
    st.subheader(f"🚨 Critical Alerts ({len(critical_machines)})")
    for _, row in critical_machines.iterrows():
        clicked = render_critical_alert(row["name"], row["machine_id"], row["rul_days"], row["health_score"])
        if clicked:
            st.session_state.selected_machine = row["machine_id"]
            st.switch_page("pages/2_Machine_Analysis.py")
    st.divider()

# ── Machine Status Table ──────────────────────────────────────────────────────
st.subheader("Machine Fleet Status")

display_df = df[["machine_id", "name", "machine_type", "location", "status", "health_score", "rul_days", "last_updated"]].copy()
display_df["last_updated"] = pd.to_datetime(display_df["last_updated"]).dt.strftime("%Y-%m-%d %H:%M")
display_df["health_score"] = display_df["health_score"].round(1)
display_df["rul_days"] = display_df["rul_days"].round(0).astype(int)
display_df.columns = ["ID", "Name", "Type", "Location", "Status", "Health %", "RUL (days)", "Last Updated"]

# Color-code the Status column via styler
def color_status(val):
    color = STATUS_COLORS.get(val, "#888")
    return f"color: {color}; font-weight: 600"

styled = display_df.style.applymap(color_status, subset=["Status"])

# Render table
st.dataframe(styled, use_container_width=True, hide_index=True, height=360)

# Quick-navigate row selection hint
st.caption("To deep-dive into a machine, click **Analyze →** in the Critical Alerts section, or select a machine on the Machine Analysis page.")

st.divider()

# ── RUL Forecast Chart (next 14 days) ────────────────────────────────────────
st.subheader("RUL Forecast — Next 14 Days")
st.caption("Projected Remaining Useful Life for Warning and Critical machines")

forecast_days = 14
today = datetime.now()

fig = go.Figure()

forecast_machines = all_df[all_df["status"].isin(["Warning", "Critical"])].sort_values("rul_days")
for _, row in forecast_machines.iterrows():
    start_rul = row["rul_days"]
    # Simple linear degradation model for forecast
    daily_rate = start_rul / max(start_rul * 1.5, 30)  # rough daily decay rate
    rul_forecast = [max(0, start_rul - daily_rate * d) for d in range(forecast_days + 1)]
    dates = [today + timedelta(days=d) for d in range(forecast_days + 1)]

    color = STATUS_COLORS[row["status"]]
    fig.add_trace(go.Scatter(
        x=dates,
        y=rul_forecast,
        name=f"{row['machine_id']}",
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=4),
        hovertemplate=f"<b>{row['name']}</b><br>Date: %{{x|%b %d}}<br>RUL: %{{y:.0f}} days<extra></extra>",
    ))

# Warning threshold line
fig.add_hline(y=20, line_dash="dash", line_color="#E74C3C", annotation_text="Critical threshold (20 days)", annotation_position="bottom right")
fig.add_hline(y=60, line_dash="dot", line_color="#F39C12", annotation_text="Warning threshold (60 days)", annotation_position="bottom right")

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#1A1F2E",
    paper_bgcolor="#1A1F2E",
    height=350,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(title="Date", gridcolor="#2A2F3E"),
    yaxis=dict(title="Remaining Useful Life (days)", gridcolor="#2A2F3E"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

# ── Healthy Machines Summary ──────────────────────────────────────────────────
st.divider()
st.subheader("✅ Healthy Fleet Overview")
healthy_df = all_df[all_df["status"] == "Healthy"][["machine_id", "name", "location", "health_score", "rul_days"]].copy()
healthy_df.columns = ["ID", "Name", "Location", "Health %", "RUL (days)"]
healthy_df["Health %"] = healthy_df["Health %"].round(1)
healthy_df["RUL (days)"] = healthy_df["RUL (days)"].round(0).astype(int)
st.dataframe(healthy_df, use_container_width=True, hide_index=True)
