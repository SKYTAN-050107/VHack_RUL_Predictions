import streamlit as st
import requests
import pandas as pd

# API Base URL
API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Maintenance Platform - VHACK", page_icon="🛠️", layout="wide")

# Check authentication
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.title("Access Denied")
    st.warning("Please log in on the [Home Page](../app.py) to access this platform.")
    st.stop()

st.title("🛠️ Maintenance Operations Platform")

# Tabs for Work Orders and Staff Management
tab_orders, tab_staff = st.tabs(["📋 Active Work Orders", "👥 Staff Management"])

with tab_staff:
    st.subheader("Technician Management")
    
    # Helper to call API for staff
    def get_staff():
        try:
            resp = requests.get(f"{API_URL}/maintenance/staff")
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return []

    staff_data = get_staff()
    
    if staff_data:
        df_staff = pd.DataFrame(staff_data)
        st.dataframe(df_staff, use_container_width=True)
        
        # Add Delete Staff functionality
        with st.expander("🗑️ De-register Technician"):
            staff_to_delete = st.selectbox("Select Staff to Remove", [s["name"] for s in staff_data])
            if st.button("Confirm Removal", type="secondary"):
                s_id = next(s["id"] for s in staff_data if s["name"] == staff_to_delete)
                resp = requests.delete(f"{API_URL}/maintenance/staff/{s_id}")
                if resp.status_code == 200:
                    st.success(f"Technician {staff_to_delete} removed from the roster.")
                    st.rerun()
    
    # Simple Staff CRUD Form
    with st.expander("➕ Add New Technician"):
        with st.form("add_staff_form"):
            new_name = st.text_input("Full Name")
            new_role = st.selectbox("Role", ["Senior Technician", "Junior Technician", "Maintenance Manager"])
            new_specialty = st.selectbox("Specialty", ["Mechanical", "Electrical", "Software", "Generalist"])
            
            if st.form_submit_button("Register Staff"):
                if new_name:
                    payload = {"name": new_name, "role": new_role, "specialty": new_specialty, "status": "Available"}
                    resp = requests.post(f"{API_URL}/maintenance/staff/create", json=payload)
                    if resp.status_code == 200:
                        st.success(f"{new_name} added to the maintenance roster.")
                        st.rerun()
                else:
                    st.error("Name is required.")

with tab_orders:
    st.markdown("""
    ### Step 8: Active Monitoring & Technician Execution
    Welcome to the real-time maintenance hub. This platform tracks all authorized work orders, providing technicians with the technical grounding needed for precision repairs.
    """)

# Helper to call API
def get_active_maintenance():
    try:
        response = requests.get(f"{API_URL}/maintenance/active")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Backend connection error: {e}")
    return []

active_tasks = get_active_maintenance()

# Display active maintenance
if not active_tasks:
    st.info("✅ All systems operational. No active maintenance tasks in the queue.")
else:
    st.subheader("📋 Active Work Orders")
    
    for task in active_tasks:
        with st.expander(f"Task #{task['id']}: {task['action_taken']} - Machine ID: {task['machine_id']}"):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown("#### 🔧 Technical Instructions")
                st.write(f"**AI Predicted Cause:** *{task.get('root_cause_prediction', 'N/A')}*")
                st.write("**Detailed Steps:**")
                steps = task.get("steps", [])
                if isinstance(steps, list):
                    for step in steps:
                        st.write(f"- {step}")
                else:
                    st.write(steps)
                
            with col2:
                st.markdown("#### 📦 Required Components")
                components = task.get("components", [])
                if isinstance(components, list):
                    for comp in components:
                        if isinstance(comp, dict):
                            st.write(f"- **{comp.get('name', 'Unknown')}** (Qty: {comp.get('quantity', 'N/A')})")
                        else:
                            st.write(f"- {comp}")
                else:
                    st.write(components)
                
            with col3:
                st.markdown("#### 👤 Assignment")
                st.info(f"**Technician:** {task.get('technician_name', 'Unassigned')}")
                st.write(f"⏱ **Est. Time:** {task.get('estimated_time', 'N/A')} hours")

            st.divider()
            
            # Step 9 & 10: Technician Report & Feedback Loop
            st.markdown("### Step 9 & 10: Service Report & Feedback Loop")
            with st.form(key=f"complete_form_{task['id']}"):
                st.write("Complete the following fields to close the work order and update the AI feedback loop.")
                
                final_root_cause = st.text_area(
                    "Verified Root Cause", 
                    placeholder="e.g., Confirmed Tier 2 bearing fatigue due to lubrication blockage.",
                    help="This data is used to improve future AI diagnostic accuracy."
                )
                
                final_action = st.text_area(
                    "Service Actions Performed",
                    placeholder="e.g., Flushed system, replaced bearings, and performed 1h QA test."
                )
                
                c1, c2 = st.columns(2)
                with c1:
                    actual_time = st.number_input("Actual Time (Hours)", min_value=0.1, value=float(task.get('estimated_time', 1.0)))
                with c2:
                    parts_used = st.multiselect("Parts Replaced", ["Bearings", "Seal Kit", "Sensor", "Filter", "Lubricant"])

                if st.form_submit_button("✅ Finalize & Notify Management", type="primary", use_container_width=True):
                    if not final_root_cause or not final_action:
                        st.error("Please provide both the root cause and actions taken for the feedback loop.")
                    else:
                        with st.spinner("Closing work order and updating system..."):
                            params = {
                                "maintenance_id": task['id'],
                                "root_cause": final_root_cause,
                                "action_taken": final_action
                            }
                            resp = requests.post(f"{API_URL}/maintenance/complete/{task['id']}", params=params)
                            if resp.status_code == 200:
                                st.success(f"Work Order #{task['id']} closed. Machine RUL has been recalibrated.")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"Error closing work order: {resp.text}")

st.divider()
st.caption("VHACK Predictive Maintenance Platform | Integrated Industry 4.0 Workflow")
