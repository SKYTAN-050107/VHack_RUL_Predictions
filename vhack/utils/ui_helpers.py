"""Reusable UI components used across multiple pages."""

import streamlit as st
from config.settings import STATUS_COLORS, STATUS_ICONS


def status_badge(status: str) -> str:
    """Returns an HTML badge string for the given status."""
    color = STATUS_COLORS.get(status, "#888")
    icon = STATUS_ICONS.get(status, "⚪")
    return (
        f'<span style="background:{color};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-size:0.8em;font-weight:600;">'
        f"{icon} {status}</span>"
    )


def render_status_badge(status: str):
    st.markdown(status_badge(status), unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def render_kpi_row(metrics: list):
    """
    metrics: list of dicts with keys: label, value, delta (optional), help (optional)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
                help=m.get("help"),
            )


def role_gate(allowed_roles: list, message: str = "You do not have permission to view this section."):
    """Context manager — shows a warning and returns False if role not allowed."""
    from utils.auth import check_role
    return check_role(allowed_roles)


def render_info_box(title: str, content: str, icon: str = "ℹ️"):
    st.info(f"**{icon} {title}**\n\n{content}")


def render_warning_box(title: str, content: str):
    st.warning(f"**⚠️ {title}**\n\n{content}")


def render_critical_alert(machine_name: str, machine_id: str, rul_days: float, health_score: float):
    """Render a critical alert card. Returns True if the Analyze button was clicked."""
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"🔴 **{machine_name}** `{machine_id}`")
            st.caption(f"Health: {health_score:.1f}% | RUL: {rul_days:.0f} days")
        with col2:
            st.markdown(
                f"<span style='color:#E74C3C;font-weight:600;font-size:1.1em'>"
                f"⏰ {rul_days:.0f} days remaining</span>",
                unsafe_allow_html=True,
            )
        with col3:
            return st.button("Analyze →", key=f"alert_btn_{machine_id}", type="primary")
    return False
