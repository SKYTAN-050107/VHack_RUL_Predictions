import streamlit as st
import requests

# API Base URL
API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Input Resources - VHACK", page_icon="📄", layout="wide")

# Check authentication
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.title("Access Denied")
    st.warning("Please log in on the [Home Page](../app.py) to access this tool.")
    st.stop()

st.title("Input Resources")
st.markdown("""
Please upload the technical manuals and financial reports to provide context for AI-driven 
root cause analysis and cost estimation.
""")

# Checklist to track uploads
st.subheader("Required Documents Checklist")

# Default values
tech_uploaded = False
fin_uploaded = False

try:
    status_resp = requests.get(f"{API_URL}/resources/status")
    if status_resp.status_code == 200:
        status = status_resp.json()
        tech_uploaded = status.get("technical_manuals_uploaded", False)
        fin_uploaded = status.get("financial_reports_uploaded", False)
except Exception as e:
    st.error(f"Failed to connect to backend for status check: {e}")

st.checkbox("Technical Manuals Uploaded", value=tech_uploaded, disabled=True)
st.checkbox("Financial Reports Uploaded", value=fin_uploaded, disabled=True)

# Upload boxes
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Technical Resource")
    tech_file = st.file_uploader("Drag and drop PDF or TXT here", type=["pdf", "txt", "md"], key="tech_uploader")
    if tech_file:
        if st.button("Upload Technical Manual"):
            with st.spinner("Processing file for RAG..."):
                file_type = "application/pdf" if tech_file.name.endswith(".pdf") else "text/plain"
                files = {"file": (tech_file.name, tech_file.getvalue(), file_type)}
                data = {"resource_type": "technical"}
                resp = requests.post(f"{API_URL}/resources/upload", files=files, data=data)
                if resp.status_code == 200:
                    st.success("Technical manual processed and stored in vector database!")
                    st.rerun()
                else:
                    st.error(f"Failed to upload: {resp.text}")

with col2:
    st.subheader("2. Financial Report")
    fin_file = st.file_uploader("Drag and drop PDF or TXT here", type=["pdf", "txt", "md"], key="fin_uploader")
    if fin_file:
        if st.button("Upload Financial Report"):
            with st.spinner("Processing file for RAG..."):
                file_type = "application/pdf" if fin_file.name.endswith(".pdf") else "text/plain"
                files = {"file": (fin_file.name, fin_file.getvalue(), file_type)}
                data = {"resource_type": "financial"}
                resp = requests.post(f"{API_URL}/resources/upload", files=files, data=data)
                if resp.status_code == 200:
                    st.success("Financial report processed and stored in vector database!")
                    st.rerun()
                else:
                    st.error(f"Failed to upload: {resp.text}")

st.info("Uploaded files are converted and put into the vector database for RAG (Retrieval-Augmented Generation).")
