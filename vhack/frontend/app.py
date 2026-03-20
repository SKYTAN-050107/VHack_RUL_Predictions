import streamlit as st
import requests

# API Base URL
API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="VHACK Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
)

# Initialize session state for auth
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

def login_user(email, password):
    try:
        resp = requests.post(f"{API_URL}/auth/login", json={"email": email, "password": password})
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.session_state.token = data["access_token"]
            return True
        else:
            st.error(f"Login failed: {resp.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return False

def signup_user(email, password):
    try:
        resp = requests.post(f"{API_URL}/auth/signup", json={"email": email, "password": password})
        if resp.status_code == 200:
            st.success("Signup successful! Please check your email for confirmation (if enabled).")
            return True
        else:
            st.error(f"Signup failed: {resp.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return False

# Sidebar for Login/Logout
st.sidebar.title("Authentication")

if not st.session_state.authenticated:
    tab1, tab2 = st.sidebar.tabs(["Login", "Sign Up"])
    
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_user(email, password):
                st.rerun()
                
    with tab2:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            signup_user(new_email, new_password)
else:
    st.sidebar.write(f"Logged in as: **{st.session_state.user_email}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.token = None
        st.rerun()

# Main Content
if not st.session_state.authenticated:
    st.title("Predictive Maintenance for SME Resilience")
    st.warning("Please log in to access the dashboard and tools.")
    st.markdown("""
    This platform helps Small and Medium Enterprises (SMEs) proactively manage machinery health using AI-driven insights.
    
    ### Key Features:
    - **Real-time Machine Monitoring**: Track RUL and health indicators.
    - **AI Root Cause Analysis**: Understand why machinery is failing.
    - **Maintenance Planning**: Approve work orders and generate technical instructions.
    - **RAG-powered Knowledge Base**: Upload manuals and reports for grounded AI advice.
    """)
else:
    st.title("Welcome to the Maintenance Dashboard")
    st.success(f"Hello, {st.session_state.user_email}! You have full access to the platform.")
    st.markdown("""
    Use the navigation menu on the left to explore:
    - **Overview**: Check machine status and AI analysis.
    - **Input Resource**: Upload technical manuals and financial reports.
    - **Input ML Data**: Upload sensor CSVs for RUL prediction.
    - **Maintenance Platform**: Manage work orders and technician reports.
    """)

st.sidebar.divider()
st.sidebar.title("Navigation")
st.sidebar.info("Select a page above to get started.")
