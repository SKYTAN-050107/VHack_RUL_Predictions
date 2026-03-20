import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Input ML Data", layout="wide")

API_URL = "http://localhost:8000/api"


def get_machines():
    try:
        response = requests.get(f"{API_URL}/machines/")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def get_model_catalog():
    fallback_ids = ["FD001", "FD002", "FD003", "FD004"]
    fallback_readiness = {
        dataset_id: {
            "available": False,
            "message": "Model availability unknown",
        }
        for dataset_id in fallback_ids
    }
    try:
        response = requests.get(f"{API_URL}/machines/models/options", timeout=10)
        if response.status_code == 200:
            payload = response.json()
            options = payload.get("dataset_ids", [])
            if options:
                return {
                    "dataset_ids": options,
                    "readiness": payload.get("readiness", fallback_readiness),
                    "ready_dataset_ids": payload.get("ready_dataset_ids", []),
                }
    except Exception:
        pass
    return {
        "dataset_ids": fallback_ids,
        "readiness": fallback_readiness,
        "ready_dataset_ids": [],
    }

st.title("📈 Input Machinery ML Data")

st.write("Upload sensor data in CSV format (NASA CMAPSS standard) for RUL prediction.")

machines = get_machines()
if not machines:
    st.error("No machines found. Start backend and initialize machine records first.")
    st.stop()

machine_labels = [f"{m['id']} - {m['name']}" for m in machines]
selected_machine_label = st.selectbox("Select Machine", machine_labels)
selected_machine_id = selected_machine_label.split(" - ")[0]
model_catalog = get_model_catalog()
dataset_options = model_catalog.get("dataset_ids", ["FD001", "FD002", "FD003", "FD004"])
selected_dataset = st.selectbox("Select Model Dataset", dataset_options, index=0)
readiness = model_catalog.get("readiness", {})
selected_readiness = readiness.get(selected_dataset, {})
selected_available = bool(selected_readiness.get("available", False))

if selected_available:
    st.success(f"{selected_dataset} model is ready for inference.")
else:
    st.warning(
        f"{selected_dataset} model is not ready: "
        f"{selected_readiness.get('message', 'Model artifact not found')}"
    )
    st.caption("Upload preview still works, but prediction run is disabled until a model is available.")

uploaded_file = st.file_uploader("Drag and drop sensor data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.dataframe(df.head())
    
    st.info("Detected Columns: " + ", ".join(df.columns))
    
    if st.button("Run RUL Prediction Model", type="primary", disabled=not selected_available):
        with st.spinner("ML Model inferencing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(
                f"{API_URL}/machines/{selected_machine_id}/upload-sensor-data",
                files=files,
                params={"dataset_id": selected_dataset},
                timeout=30,
            )

        if response.status_code == 200:
            prediction = response.json().get("prediction", {})
            st.success(
                "Prediction complete. "
                f"Estimated Remaining Useful Life: {prediction.get('predicted_rul', 'N/A')} cycles."
            )
            st.write(f"**Status:** {prediction.get('status', 'Unknown')}")
            st.write(f"**Health State:** {prediction.get('health_state', 'Unknown')}")
            st.write(f"**Dataset Used:** {prediction.get('dataset_id', selected_dataset)}")
            if prediction.get("change_point_detected"):
                cp_step = prediction.get("change_point_step")
                if cp_step is not None:
                    st.warning(f"Change point detected at cycle index {cp_step}.")
                else:
                    st.warning("Change point detected in recent readings.")
            explanation = prediction.get("explanation")
            if explanation:
                st.info(explanation)
        else:
            st.error(f"Prediction failed: {response.text}")

st.divider()
st.subheader("Data Format Guide")
st.markdown("""
Your CSV should follow the NASA CMAPSS format:
- **Unit ID**: Machine identifier
- **Cycle**: Current operation cycle
- **Sensor 1-21**: Vibration, temperature, load, etc.

MVP validation rules:
- At least 3 rows of sensor readings
- Numeric sensor values required
- Optional identifier columns (unit_id, cycle) are ignored during inference
""")
