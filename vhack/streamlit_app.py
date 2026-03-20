"""
Main entry point — shows the login page.
After successful login, redirects to the Dashboard.
"""

import streamlit as st
from utils.auth import init_session_state, login_page

st.set_page_config(
    page_title="Digital Machinery Caretaker",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_session_state()

if st.session_state.get("authenticated", False):
    st.switch_page("pages/1_Dashboard.py")
else:
    login_page()
