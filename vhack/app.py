import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance for SME",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Predictive Maintenance Dashboard")

st.markdown("""
### Welcome to the SME Resilience Platform
This platform helps Small and Medium Enterprises (SMEs) monitor industrial machinery health, 
predict failures, and manage maintenance tasks efficiently.

**Get started by selecting a page from the sidebar.**
""")

with st.sidebar:
    st.info("💡 **SDG 9: Industry, Innovation, and Infrastructure**")
    st.success("Target 9.4: Upgrade infrastructure and retrofit industries for sustainability.")
