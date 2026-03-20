import streamlit as st
import pandas as pd
from datetime import datetime

from utils.auth import require_login, render_sidebar_user, check_role
from utils.mock_api_client import MockMLBackendClient
from utils.gemini_client import GeminiClient
from utils.ui_helpers import render_page_header
from config.settings import PLAN_COLORS, STATUS_COLORS

st.set_page_config(page_title="Maintenance Plans | DMC", page_icon="🗂️", layout="wide")
require_login()

# ── Sidebar ───────────────────────────────────────────────────────────────────
client = MockMLBackendClient()
machines = client.fetch_all_machines()
machine_ids = [m.machine_id for m in machines]
machine_labels = {m.machine_id: f"{m.machine_id} — {m.name}" for m in machines}

with st.sidebar:
    st.markdown("## 🗂️ Maintenance Plans")
    st.divider()
    pre_selected = st.session_state.get("selected_machine")
    default_idx = machine_ids.index(pre_selected) if pre_selected in machine_ids else 0
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
is_manager = check_role(["Manager", "Admin"])

render_page_header("🗂️ Maintenance Decision Support", f"{machine.name} | RUL: {pred.rul_days:.0f} days | Health: {pred.health_score:.1f}%")

# Machine status summary
status_color = STATUS_COLORS.get(machine.status, "#888")
st.markdown(
    f"<div style='background:{status_color}22;border-left:4px solid {status_color};"
    f"padding:12px 16px;border-radius:4px;margin-bottom:16px'>"
    f"<strong style='color:{status_color}'>{machine.status}</strong> — "
    f"{pred.predicted_failure_mode}</div>",
    unsafe_allow_html=True,
)

st.info(f"**Financial impact if failure:** See the Financial Risk page for full cost breakdown. "
        f"Estimated downtime cost for a {machine.machine_type} at this degradation level: $45,000–$150,000.")

# ── Generate Plans Button ─────────────────────────────────────────────────────
plans_key = f"plans_{selected_id}"
col_btn, col_hint = st.columns([1, 3])
with col_btn:
    gen_plans = st.button("Generate 3 Maintenance Plans", type="primary", icon="🤖")
with col_hint:
    st.caption("AI generates URGENT, BALANCED, and DEFERRED strategies tailored to this machine's condition.")

if gen_plans:
    with st.spinner("Generating maintenance strategies with AI..."):
        gemini = GeminiClient()
        financial_impact = 85000.0  # Default estimate — real value comes from Financial Risk page
        fin_saved = st.session_state.get("financial_inputs", {}).get(selected_id, {})
        if fin_saved:
            hourly = fin_saved.get("hourly_production_value", 10000)
            mttr   = fin_saved.get("mttr_hours", 8)
            repair = fin_saved.get("repair_cost_failure", 20000)
            sla    = fin_saved.get("sla_penalty_per_hour", 1000) * mttr
            sc     = fin_saved.get("supply_chain_penalty", 3000)
            financial_impact = hourly * mttr + repair + sla + sc

        plans_data = gemini.generate_maintenance_plans(selected_id, machine, pred, financial_impact)
        st.session_state.generated_plans[plans_key] = plans_data
    st.rerun()

# ── Display Plans ─────────────────────────────────────────────────────────────
if plans_key not in st.session_state.get("generated_plans", {}):
    st.info("Click **Generate 3 Maintenance Plans** to create AI-powered maintenance strategies for this machine.")
    st.stop()

plans_data = st.session_state.generated_plans[plans_key]
plans = plans_data.get("plans", [])

if not plans:
    st.error("Could not parse maintenance plans. Please try regenerating.")
    st.stop()

plan_labels = {
    "URGENT":   "🚨 URGENT",
    "BALANCED": "⚖️ BALANCED",
    "DEFERRED": "📅 DEFERRED",
}

tab_urgent, tab_balanced, tab_deferred = st.tabs(
    [plan_labels.get(p["plan_type"], p["plan_type"]) for p in plans[:3]]
)

tabs = [tab_urgent, tab_balanced, tab_deferred]

for tab, plan in zip(tabs, plans[:3]):
    plan_type = plan["plan_type"]
    color = PLAN_COLORS.get(plan_type, "#888")

    with tab:
        # Plan headline banner
        st.markdown(
            f"<div style='background:{color}22;border-left:4px solid {color};"
            f"padding:10px 16px;border-radius:4px;margin-bottom:12px'>"
            f"<strong style='color:{color};font-size:1.05em'>{plan.get('headline','')}</strong></div>",
            unsafe_allow_html=True,
        )

        # Metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        timeline_h = plan.get("timeline_hours", 0)
        timeline_label = f"{timeline_h}h" if timeline_h < 24 else f"{timeline_h//24}d {timeline_h%24}h"
        mc1.metric("⏱️ Timeline", timeline_label)
        mc2.metric("💵 Est. Cost", f"${plan.get('estimated_cost_usd', 0):,}")
        mc3.metric("🛡️ Risk Level", plan.get("risk_level", "—"))
        mc4.metric("✅ Success Rate", f"{plan.get('success_rate_pct', 0)}%")

        st.divider()
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            # Actions checklist
            st.markdown("**Step-by-Step Actions**")
            actions = plan.get("actions", [])
            for i, action in enumerate(actions, 1):
                st.markdown(f"{i}. {action}")

            # When to use
            if plan.get("best_for"):
                st.info(f"**💡 When to use this plan:**\n\n{plan['best_for']}")

        with col_right:
            # Parts required
            parts = plan.get("parts_required", [])
            if parts:
                st.markdown("**Parts Required**")
                parts_df = pd.DataFrame(parts)
                if "unit_cost" in parts_df.columns and "quantity" in parts_df.columns:
                    parts_df["total_cost"] = parts_df["unit_cost"] * parts_df["quantity"]
                    parts_df["unit_cost"]  = parts_df["unit_cost"].apply(lambda x: f"${x:,.0f}")
                    parts_df["total_cost"] = parts_df["total_cost"].apply(lambda x: f"${x:,.0f}")
                cols_to_show = [c for c in ["name", "part_number", "quantity", "unit_cost", "total_cost"] if c in parts_df.columns]
                parts_df.columns = [c.replace("_", " ").title() for c in parts_df.columns]
                st.dataframe(parts_df[[c.replace("_", " ").title() for c in cols_to_show]], use_container_width=True, hide_index=True)

            # Labor requirements
            labor = plan.get("labor_requirements", [])
            if labor:
                st.markdown("**Labor Requirements**")
                labor_df = pd.DataFrame(labor)
                labor_df.columns = [c.replace("_", " ").title() for c in labor_df.columns]
                st.dataframe(labor_df, use_container_width=True, hide_index=True)

        st.divider()
        col_pros, col_cons = st.columns(2)

        with col_pros:
            pros = plan.get("pros", [])
            if pros:
                st.markdown("**✅ Pros**")
                for p in pros:
                    st.markdown(f"- {p}")

        with col_cons:
            cons = plan.get("cons", [])
            if cons:
                st.markdown("**⚠️ Cons**")
                for c in cons:
                    st.markdown(f"- {c}")

        # Risk + Contingency expanders
        with st.expander("🛡️ Risk Assessment"):
            risk_level = plan.get("risk_level", "N/A")
            success = plan.get("success_rate_pct", 0)
            st.markdown(
                f"- **Risk Level:** {risk_level}\n"
                f"- **Success Rate:** {success}%\n"
                f"- **Failure probability if plan not executed:** Based on current RUL of {pred.rul_days:.0f} days and {machine.status} status"
            )

        with st.expander("🔄 Contingency Plan"):
            contingency = plan.get("contingency_plan", "No contingency plan specified.")
            st.markdown(contingency)

        st.divider()

        # Decision section — Managers/Admins only
        if is_manager:
            st.markdown("**📋 Decision**")
            decision_col1, decision_col2 = st.columns(2)

            with decision_col1:
                approval_key = f"approved_{selected_id}_{plan_type}"
                if approval_key in [a.get("key") for a in st.session_state.get("approved_plans", [])]:
                    st.success(f"✅ **{plan_type} plan approved** for {machine.machine_id}")
                else:
                    if st.button(f"✅ Approve {plan_type} Plan", key=f"approve_{selected_id}_{plan_type}", type="primary", use_container_width=True):
                        approval = {
                            "key": approval_key,
                            "machine_id": selected_id,
                            "machine_name": machine.name,
                            "plan_type": plan_type,
                            "approved_by": st.session_state.get("user_name", "Unknown"),
                            "approved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "estimated_cost": plan.get("estimated_cost_usd", 0),
                        }
                        st.session_state.approved_plans.append(approval)
                        st.rerun()

            with decision_col2:
                # Export plan as text
                plan_md = (
                    f"# Maintenance Plan: {plan_type} — {machine.name}\n\n"
                    f"**Machine:** {selected_id} | **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                    f"**Headline:** {plan.get('headline', '')}\n\n"
                    f"| Metric | Value |\n|---|---|\n"
                    f"| Timeline | {timeline_label} |\n"
                    f"| Cost | ${plan.get('estimated_cost_usd', 0):,} |\n"
                    f"| Risk Level | {plan.get('risk_level', '')} |\n"
                    f"| Success Rate | {plan.get('success_rate_pct', 0)}% |\n\n"
                    f"## Actions\n\n"
                    + "\n".join(f"{i+1}. {a}" for i, a in enumerate(plan.get("actions", [])))
                    + f"\n\n## Contingency\n\n{plan.get('contingency_plan', '')}\n"
                )
                st.download_button(
                    f"⬇️ Download {plan_type} Plan",
                    data=plan_md,
                    file_name=f"maintenance_plan_{selected_id}_{plan_type.lower()}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        else:
            st.caption("Log in as Manager or Admin to approve maintenance plans.")

# ── Approved Plans Log ────────────────────────────────────────────────────────
approved = st.session_state.get("approved_plans", [])
if approved:
    st.divider()
    st.subheader("📋 Approved Plans Log (this session)")
    approved_df = pd.DataFrame(approved)[["machine_id", "machine_name", "plan_type", "approved_by", "approved_at", "estimated_cost"]]
    approved_df["estimated_cost"] = approved_df["estimated_cost"].apply(lambda x: f"${x:,}")
    approved_df.columns = ["Machine ID", "Machine Name", "Plan Type", "Approved By", "Approved At", "Est. Cost"]
    st.dataframe(approved_df, use_container_width=True, hide_index=True)
