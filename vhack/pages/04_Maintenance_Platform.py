import streamlit as st
import pandas as pd

st.set_page_config(page_title="Maintenance Platform", layout="wide")

st.title("🔧 Maintenance Platform")

# Mock maintenance tasks
if "tasks" not in st.session_state:
    st.session_state.tasks = [
        {"id": 1, "machine": "Conveyor-C3", "technician": "John Doe", "status": "Ongoing", "date": "2026-03-14"},
        {"id": 2, "machine": "Pump-B2", "technician": "Sarah Smith", "status": "Scheduled", "date": "2026-03-16"},
    ]

st.subheader("📋 Ongoing Maintenance Plans")

for task in st.session_state.tasks:
    with st.expander(f"{task['machine']} - Technician: {task['technician']} ({task['status']})"):
        st.write(f"**Date:** {task['date']}")
        
        st.markdown("---")
        st.subheader("🛠️ Maintenance Details")
        st.write("**Root Cause Analysis:** Bearings nearing end-of-life due to fatigue.")
        st.write("**Steps Taken:**")
        st.write("1. Power down and lock-out machine.")
        st.write("2. Remove protective casing.")
        st.write("3. Inspect bearing for physical damage.")
        
        if task['status'] == "Ongoing":
            st.markdown("---")
            st.subheader("📝 Maintenance Completion Form")
            with st.form(f"complete_form_{task['id']}"):
                actual_cause = st.text_area("Actual failure cause found:")
                parts_used = st.text_input("Parts replaced:")
                labor_hours = st.number_input("Labor hours:", min_value=0.5, step=0.5)
                
                submitted = st.form_submit_button("Complete Maintenance")
                if submitted:
                    task['status'] = "Completed"
                    st.success("Maintenance log updated! RUL will be reset for this machine.")
                    # In a real app, update DB and reset RUL in Overview
                    st.rerun()
