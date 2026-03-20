import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils.auth import require_login, render_sidebar_user
from utils.ui_helpers import render_page_header
from data.mock_schedules import MOCK_SCHEDULES, TEAM_CAPACITY
from config.settings import SCHEDULE_STATUS_COLORS

st.set_page_config(page_title="Schedule | DMC", page_icon="📅", layout="wide")
require_login()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📅 Maintenance Schedule")
    st.divider()
    st.markdown("**Filters**")
    status_filter = st.multiselect(
        "Schedule Status",
        ["Urgent", "Scheduled", "Pending", "Completed"],
        default=["Urgent", "Scheduled", "Pending"],
    )
    team_filter = st.multiselect(
        "Team",
        ["Team Alpha", "Team Beta", "Team Gamma"],
        default=["Team Alpha", "Team Beta", "Team Gamma"],
    )

render_sidebar_user()

render_page_header("📅 Maintenance Schedule", "Next 30 days — all maintenance events and resource allocation")

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(MOCK_SCHEDULES)

# Apply filters
if status_filter:
    df = df[df["status"].isin(status_filter)]
if team_filter:
    df = df[df["assigned_team"].isin(team_filter)]

df["start"]  = pd.to_datetime(df["start"])
df["finish"] = pd.to_datetime(df["finish"])
df["recommended_date"] = pd.to_datetime(df["recommended_date"])

# ── Summary KPIs ──────────────────────────────────────────────────────────────
all_df = pd.DataFrame(MOCK_SCHEDULES)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Scheduled", len(all_df))
with c2:
    urgent_count = int((all_df["status"] == "Urgent").sum())
    st.metric("🔴 Urgent", urgent_count)
with c3:
    scheduled_count = int((all_df["status"] == "Scheduled").sum())
    st.metric("🔵 Scheduled", scheduled_count)
with c4:
    total_hours = int(all_df["duration_hours"].sum())
    st.metric("⏱️ Total Hours", f"{total_hours}h")

st.divider()

# ── Gantt Chart ───────────────────────────────────────────────────────────────
st.subheader("📊 Maintenance Gantt Chart")

if df.empty:
    st.info("No maintenance events match the current filters.")
else:
    gantt_df = df[["machine_name", "start", "finish", "status", "assigned_team", "description", "plan_type", "duration_hours"]].copy()
    gantt_df.columns = ["Machine", "Start", "Finish", "Status", "Team", "Description", "Plan", "Duration (hrs)"]

    fig_gantt = px.timeline(
        gantt_df,
        x_start="Start",
        x_end="Finish",
        y="Machine",
        color="Status",
        color_discrete_map=SCHEDULE_STATUS_COLORS,
        custom_data=["Team", "Description", "Plan", "Duration (hrs)"],
        title="",
    )

    fig_gantt.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Start: %{base|%Y-%m-%d %H:%M}<br>"
            "End: %{x|%Y-%m-%d %H:%M}<br>"
            "Team: %{customdata[0]}<br>"
            "Plan: %{customdata[2]}<br>"
            "Duration: %{customdata[3]}h<br>"
            "<i>%{customdata[1]}</i><extra></extra>"
        )
    )

    # Today marker
    today = datetime.now()
    fig_gantt.add_vline(x=today, line_dash="dash", line_color="#FAFAFA", opacity=0.6,
                        annotation_text="Today", annotation_position="top")

    fig_gantt.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1A1F2E",
        paper_bgcolor="#1A1F2E",
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Date", gridcolor="#2A2F3E"),
        yaxis=dict(title="", autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig_gantt, use_container_width=True)

st.divider()

# ── Schedule Detail Table ─────────────────────────────────────────────────────
st.subheader("📋 Schedule Detail")

if not df.empty:
    display_df = df[["schedule_id", "machine_id", "machine_name", "status", "recommended_date",
                      "duration_hours", "assigned_team", "technician", "plan_type", "parts_ordered"]].copy()
    display_df["recommended_date"] = display_df["recommended_date"].dt.strftime("%Y-%m-%d")
    display_df["parts_ordered"] = display_df["parts_ordered"].map({True: "✅ Yes", False: "⏳ Pending"})
    display_df.columns = ["ID", "Machine ID", "Machine Name", "Status", "Date", "Duration (h)", "Team", "Technician", "Plan", "Parts"]

    def color_sched_status(val):
        return f"color: {SCHEDULE_STATUS_COLORS.get(val, '#888')}; font-weight: 600"

    styled_sched = display_df.style.applymap(color_sched_status, subset=["Status"])
    st.dataframe(styled_sched, use_container_width=True, hide_index=True)
else:
    st.info("No events match the current filters.")

st.divider()

# ── Resource Utilization Chart ────────────────────────────────────────────────
st.subheader("👥 Resource Utilization")

all_sched_df = pd.DataFrame(MOCK_SCHEDULES)
team_hours = all_sched_df.groupby("assigned_team")["duration_hours"].sum().reset_index()
team_hours.columns = ["Team", "Scheduled Hours"]

util_rows = []
for _, row in team_hours.iterrows():
    team = row["Team"]
    capacity = TEAM_CAPACITY.get(team, {})
    available_weekly = capacity.get("available_hours_per_week", 40)
    # Scale to 4-week window
    available_4wk = available_weekly * 4
    scheduled     = float(row["Scheduled Hours"])
    utilization   = min(scheduled / available_4wk * 100, 100)
    util_rows.append({
        "Team": team,
        "Scheduled (h)": scheduled,
        "Available 4-wk (h)": available_4wk,
        "Utilization (%)": round(utilization, 1),
        "Specialty": capacity.get("specialty", ""),
    })

util_df = pd.DataFrame(util_rows)

# Horizontal bar chart
fig_util = go.Figure()
fig_util.add_trace(go.Bar(
    x=util_df["Utilization (%)"],
    y=util_df["Team"],
    orientation="h",
    marker=dict(
        color=[
            "#E74C3C" if u > 80 else "#F39C12" if u > 60 else "#2ECC71"
            for u in util_df["Utilization (%)"]
        ]
    ),
    text=[f"{u}%   ({s:.0f}/{a:.0f}h)" for u, s, a in zip(util_df["Utilization (%)"], util_df["Scheduled (h)"], util_df["Available 4-wk (h)"])],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Utilization: %{x:.1f}%<br>Scheduled: %{customdata[0]}h<br>Available: %{customdata[1]}h<extra></extra>",
    customdata=list(zip(util_df["Scheduled (h)"], util_df["Available 4-wk (h)"])),
))

fig_util.add_vline(x=80, line_dash="dash", line_color="#E74C3C", annotation_text="High load (80%)", annotation_position="top right")
fig_util.add_vline(x=60, line_dash="dot",  line_color="#F39C12", annotation_text="Medium load (60%)", annotation_position="top right")

fig_util.update_layout(
    template="plotly_dark",
    plot_bgcolor="#1A1F2E",
    paper_bgcolor="#1A1F2E",
    height=250,
    margin=dict(l=10, r=100, t=20, b=10),
    xaxis=dict(title="Utilization (%)", range=[0, 120], gridcolor="#2A2F3E"),
    yaxis=dict(title=""),
    showlegend=False,
)
st.plotly_chart(fig_util, use_container_width=True)

# Utilization detail table
util_display = util_df[["Team", "Specialty", "Scheduled (h)", "Available 4-wk (h)", "Utilization (%)"]].copy()
st.dataframe(util_display, use_container_width=True, hide_index=True)

# ── Conflict Detection ────────────────────────────────────────────────────────
st.divider()
st.subheader("⚡ Conflict Detection")

all_sched_df["start"]  = pd.to_datetime(all_sched_df["start"])
all_sched_df["finish"] = pd.to_datetime(all_sched_df["finish"])

conflicts = []
records = all_sched_df.to_dict("records")
for i, a in enumerate(records):
    for b in records[i+1:]:
        if a["assigned_team"] == b["assigned_team"]:
            # Overlap check
            if a["start"] < b["finish"] and b["start"] < a["finish"]:
                conflicts.append({
                    "Team": a["assigned_team"],
                    "Conflict": f"{a['machine_name']} ({a['start'].strftime('%m/%d %H:%M')}) overlaps with "
                                f"{b['machine_name']} ({b['start'].strftime('%m/%d %H:%M')})",
                })

if conflicts:
    st.warning(f"**{len(conflicts)} scheduling conflict(s) detected:**")
    for conflict in conflicts:
        st.markdown(f"- **{conflict['Team']}**: {conflict['Conflict']}")
else:
    st.success("✅ No scheduling conflicts detected across all teams.")
