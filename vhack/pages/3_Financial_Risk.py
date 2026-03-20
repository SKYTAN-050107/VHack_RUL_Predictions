import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

from utils.auth import require_login, render_sidebar_user, check_role
from utils.mock_api_client import MockMLBackendClient
from utils.gemini_client import GeminiClient
from utils.document_parser import DocumentParser
from utils.ui_helpers import render_page_header
from models.financial import FinancialParameters, FinancialCalculation

st.set_page_config(page_title="Financial Risk | DMC", page_icon="💰", layout="wide")
require_login()

# ── Sidebar ───────────────────────────────────────────────────────────────────
client = MockMLBackendClient()
machines = client.fetch_all_machines()
machine_ids = [m.machine_id for m in machines]
machine_labels = {m.machine_id: f"{m.machine_id} — {m.name}" for m in machines}

with st.sidebar:
    st.markdown("## 💰 Financial Risk")
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

# ── Load machine data ─────────────────────────────────────────────────────────
machine = client.fetch_machine_info(selected_id)
pred    = client.fetch_ml_prediction(selected_id)

render_page_header("💰 Financial Risk Calculator", f"{machine.name} | {machine.machine_id}")

# Role gate — Technicians see a warning; Managers/Admins see full detail
is_manager = check_role(["Manager", "Admin"])
if not is_manager:
    st.warning("⚠️ You are viewing limited financial information. Full cost breakdowns are visible to Managers and Admins only.")

# ── Input Method Toggle ───────────────────────────────────────────────────────
st.subheader("📥 Input Financial Parameters")
input_method = st.radio("Input method", ["Fill Form", "Upload Document"], horizontal=True)

params: FinancialParameters | None = None
source_label = "Manual Entry"

if input_method == "Upload Document":
    st.markdown("Upload your **SLA contract, production spec, or financial document** to auto-extract parameters.")
    uploaded = st.file_uploader(
        "Upload document (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx", "png", "jpg"],
        key="fin_upload",
    )
    if uploaded:
        with st.spinner("Extracting financial parameters from document..."):
            parser = DocumentParser()
            params = parser.parse(uploaded, selected_id)

        source_label = params.source or uploaded.name
        st.success(f"✅ Parameters extracted from: **{source_label}**")

        if is_manager:
            with st.expander("📄 Extracted Parameters (with confidence)", expanded=True):
                conf_pct = f"{params.confidence * 100:.0f}%" if params.confidence else "N/A"
                st.caption(f"Overall extraction confidence: **{conf_pct}**")
                ext_df = pd.DataFrame({
                    "Parameter": [
                        "Hourly Production Value (USD)",
                        "Units per Hour",
                        "Unit Price (USD)",
                        "Preventative Repair Cost (USD)",
                        "Failure Repair Cost (USD)",
                        "MTTR (hours)",
                        "SLA Penalty per Hour (USD)",
                        "Supply Chain Penalty (USD)",
                    ],
                    "Value": [
                        f"${params.hourly_production_value:,.0f}",
                        str(params.units_per_hour),
                        f"${params.unit_price:,.2f}",
                        f"${params.repair_cost_preventative:,.0f}",
                        f"${params.repair_cost_failure:,.0f}",
                        f"{params.mttr_hours:.1f} hrs",
                        f"${params.sla_penalty_per_hour:,.0f}",
                        f"${params.supply_chain_penalty:,.0f}",
                    ],
                    "Source": [source_label] * 8,
                })
                st.dataframe(ext_df, use_container_width=True, hide_index=True)
    else:
        st.info("Upload a document to auto-extract financial parameters, or switch to **Fill Form** mode.")

# Form input (shown when mode = "Fill Form" OR after upload to allow editing)
if input_method == "Fill Form" or (input_method == "Upload Document" and params is not None):
    header = "✏️ Edit Extracted Values" if params else "✏️ Enter Financial Parameters"
    with st.expander(header, expanded=(input_method == "Fill Form")):
        defaults = params if params else FinancialParameters(
            machine_id=selected_id,
            hourly_production_value=10000.0,
            units_per_hour=200,
            unit_price=50.0,
            repair_cost_preventative=5000.0,
            repair_cost_failure=25000.0,
            mttr_hours=8.0,
            sla_penalty_per_hour=1000.0,
            supply_chain_penalty=3000.0,
        )

        col1, col2 = st.columns(2)
        with col1:
            hourly_val   = st.number_input("Hourly Production Value (USD)", value=float(defaults.hourly_production_value), min_value=0.0, step=500.0)
            units_hr     = st.number_input("Units per Hour", value=int(defaults.units_per_hour), min_value=0, step=10)
            unit_price   = st.number_input("Unit Price (USD)", value=float(defaults.unit_price), min_value=0.0, step=1.0)
            mttr_hours   = st.number_input("MTTR — hours if failure", value=float(defaults.mttr_hours), min_value=0.5, step=0.5)
        with col2:
            repair_prev  = st.number_input("Preventative Repair Cost (USD)", value=float(defaults.repair_cost_preventative), min_value=0.0, step=500.0)
            repair_fail  = st.number_input("Failure Repair Cost (USD)", value=float(defaults.repair_cost_failure), min_value=0.0, step=500.0)
            sla_penalty  = st.number_input("SLA Penalty per Hour (USD)", value=float(defaults.sla_penalty_per_hour), min_value=0.0, step=100.0)
            sc_penalty   = st.number_input("Supply Chain Disruption Cost (USD)", value=float(defaults.supply_chain_penalty), min_value=0.0, step=500.0)

        params = FinancialParameters(
            machine_id=selected_id,
            hourly_production_value=hourly_val,
            units_per_hour=units_hr,
            unit_price=unit_price,
            repair_cost_preventative=repair_prev,
            repair_cost_failure=repair_fail,
            mttr_hours=mttr_hours,
            sla_penalty_per_hour=sla_penalty,
            supply_chain_penalty=sc_penalty,
            source=source_label,
        )
        st.session_state.financial_inputs[selected_id] = params.model_dump()

# ── Cost Calculation ──────────────────────────────────────────────────────────
if params:
    production_loss      = params.hourly_production_value * params.mttr_hours
    sla_total            = params.sla_penalty_per_hour * params.mttr_hours
    total_failure_cost   = production_loss + params.repair_cost_failure + sla_total + params.supply_chain_penalty
    total_preventative   = params.repair_cost_preventative
    net_savings          = total_failure_cost - total_preventative
    roi_pct              = (net_savings / total_preventative * 100) if total_preventative > 0 else 0

    calc = FinancialCalculation(
        machine_id=selected_id,
        production_loss=production_loss,
        total_downtime_cost_failure=total_failure_cost,
        total_preventative_cost=total_preventative,
        net_savings=net_savings,
        roi_percentage=roi_pct,
        cost_breakdown={
            "Production Loss": production_loss,
            "Failure Repair Cost": params.repair_cost_failure,
            "SLA Penalties": sla_total,
            "Supply Chain Disruption": params.supply_chain_penalty,
            "rul_days": pred.rul_days,
        },
    )

    st.divider()
    st.subheader("📊 Cost Analysis Results")

    # KPI row
    if is_manager:
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            st.metric("Total Failure Cost", f"${total_failure_cost:,.0f}", help="Total cost if machine fails unexpectedly")
        with cc2:
            st.metric("Preventative Cost", f"${total_preventative:,.0f}", help="Cost of scheduled maintenance now")
        with cc3:
            net_color = "normal" if net_savings > 0 else "inverse"
            st.metric("Net Savings", f"${net_savings:,.0f}", delta="if act now", delta_color=net_color)
        with cc4:
            st.metric("ROI", f"{roi_pct:.0f}%", help="Return on maintenance investment")
    else:
        # Technician: show simplified view
        st.info(f"Estimated cost savings from preventive maintenance: **substantial**. Contact your manager for the full financial breakdown.")

    # Cost breakdown chart — Managers only
    if is_manager:
        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            breakdown_items = {k: v for k, v in calc.cost_breakdown.items() if k != "rul_days" and v > 0}
            fig_pie = go.Figure(go.Pie(
                labels=list(breakdown_items.keys()),
                values=list(breakdown_items.values()),
                hole=0.4,
                marker=dict(colors=["#E74C3C", "#E67E22", "#F39C12", "#9B59B6"]),
                textinfo="label+percent",
                hovertemplate="%{label}<br>$%{value:,.0f}<extra></extra>",
            ))
            fig_pie.update_layout(
                title="Failure Cost Breakdown",
                template="plotly_dark", paper_bgcolor="#1A1F2E",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_table:
            rows = [
                {"Cost Component": k, "Amount (USD)": f"${v:,.0f}", "Source": params.source or "Manual Entry"}
                for k, v in breakdown_items.items()
            ]
            rows.append({"Cost Component": "**TOTAL FAILURE COST**", "Amount (USD)": f"**${total_failure_cost:,.0f}**", "Source": "—"})
            rows.append({"Cost Component": "Preventative Maintenance", "Amount (USD)": f"${total_preventative:,.0f}", "Source": params.source or "Manual Entry"})
            rows.append({"Cost Component": "**NET SAVINGS**", "Amount (USD)": f"**${net_savings:,.0f}**", "Source": "—"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Preventative vs Failure comparison bar chart
        st.markdown("#### Preventative vs. Failure Cost Comparison")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Preventative Maintenance", "Failure + Downtime"],
            y=[total_preventative, total_failure_cost],
            marker_color=["#2ECC71", "#E74C3C"],
            text=[f"${total_preventative:,.0f}", f"${total_failure_cost:,.0f}"],
            textposition="outside",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        ))
        fig_bar.update_layout(
            template="plotly_dark", plot_bgcolor="#1A1F2E", paper_bgcolor="#1A1F2E",
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Cost (USD)", gridcolor="#2A2F3E"),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Gemini Financial Recommendation ──────────────────────────────────────
    if is_manager:
        st.subheader("🤖 AI Financial Recommendation")
        rec_key = f"fin_rec_{selected_id}"
        if st.button("Generate Recommendation", type="primary", icon="🤖"):
            with st.spinner("Generating financial recommendation..."):
                gemini = GeminiClient()
                rec = gemini.generate_financial_recommendation(selected_id, params, calc)
                st.session_state.generated_analysis[rec_key] = rec
            st.rerun()

        if rec_key in st.session_state.generated_analysis:
            rec_text = st.session_state.generated_analysis[rec_key]
            # Highlight the decision line
            if "APPROVE MAINTENANCE IMMEDIATELY" in rec_text:
                st.error("**Decision: APPROVE MAINTENANCE IMMEDIATELY**", icon="🚨")
            elif "SCHEDULE WITHIN" in rec_text:
                st.warning("**Decision: SCHEDULE WITHIN 7 DAYS**", icon="⚠️")
            else:
                st.info("**Decision: MONITOR AND DEFER**", icon="📅")
            st.markdown(rec_text)
        else:
            st.info("Click **Generate Recommendation** to get an AI-powered financial decision analysis.")

        st.divider()

        # ── Export ───────────────────────────────────────────────────────────
        st.subheader("📤 Export Report")
        col_csv, col_md = st.columns(2)

        csv_data = pd.DataFrame([
            {"Machine ID": selected_id, "Machine Name": machine.name, "RUL (days)": pred.rul_days,
             "Total Failure Cost ($)": total_failure_cost, "Preventative Cost ($)": total_preventative,
             "Net Savings ($)": net_savings, "ROI (%)": round(roi_pct, 1),
             "Production Loss ($)": production_loss, "SLA Penalties ($)": sla_total,
             "Supply Chain ($)": params.supply_chain_penalty, "Source": params.source or "Manual"}
        ])

        with col_csv:
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data.to_csv(index=False),
                file_name=f"financial_risk_{selected_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        md_report = (
            f"# Financial Risk Report — {machine.name}\n\n"
            f"**Machine ID:** {selected_id}  \n"
            f"**RUL:** {pred.rul_days:.0f} days  \n"
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"## Cost Summary\n\n"
            f"| Metric | Amount |\n|---|---|\n"
            f"| Total Failure Cost | ${total_failure_cost:,.0f} |\n"
            f"| Preventative Maintenance Cost | ${total_preventative:,.0f} |\n"
            f"| **Net Savings** | **${net_savings:,.0f}** |\n"
            f"| ROI | {roi_pct:.0f}% |\n\n"
            f"## AI Recommendation\n\n"
            + st.session_state.generated_analysis.get(rec_key, "*Not yet generated.*")
        )

        with col_md:
            st.download_button(
                "⬇️ Download Markdown Report",
                data=md_report,
                file_name=f"financial_risk_{selected_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
