"""
Session-state based authentication with role gating.
Credentials are hardcoded for demo/mock mode — replace with real auth for production.
"""

import streamlit as st

_USERS = {
    "manager":    {"password": "manager123",    "role": "Manager",    "name": "Alex Chen"},
    "technician": {"password": "tech123",        "role": "Technician", "name": "Sam Rodriguez"},
    "admin":      {"password": "admin123",       "role": "Admin",      "name": "Jordan Lee"},
}

_SESSION_DEFAULTS = {
    "authenticated":       False,
    "username":            None,
    "role":                None,
    "user_name":           None,
    "selected_machine":    None,
    "financial_inputs":    {},
    "approved_plans":      [],
    "generated_analysis":  {},
    "generated_plans":     {},
}


def init_session_state():
    for key, val in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def require_login():
    """Call at the top of every page. Redirects to login if not authenticated."""
    init_session_state()
    if not st.session_state.get("authenticated", False):
        st.switch_page("streamlit_app.py")


def check_role(allowed_roles: list) -> bool:
    """Returns True if the current user's role is in allowed_roles."""
    return st.session_state.get("role", "") in allowed_roles


def login_page():
    st.markdown(
        "<h1 style='text-align:center;margin-top:80px'>⚙️ Digital Machinery Caretaker</h1>"
        "<p style='text-align:center;color:#888'>AI-Driven Predictive Maintenance Platform</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Login", use_container_width=True, type="primary")

        if submit:
            user = _USERS.get(username.lower().strip())
            if user and user["password"] == password:
                st.session_state.authenticated = True
                st.session_state.username = username.lower().strip()
                st.session_state.role = user["role"]
                st.session_state.user_name = user["name"]
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

        with st.expander("Demo credentials"):
            st.code(
                "Manager:     manager    / manager123\n"
                "Technician:  technician / tech123\n"
                "Admin:       admin      / admin123"
            )


def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("streamlit_app.py")


def render_sidebar_user():
    """Render user info + logout button in the sidebar."""
    with st.sidebar:
        st.divider()
        role = st.session_state.get("role", "")
        name = st.session_state.get("user_name", "")
        icon = {"Manager": "👔", "Technician": "🔧", "Admin": "⚡"}.get(role, "👤")
        st.markdown(f"**{icon} {name}**")
        st.caption(f"Role: {role}")
        if st.button("Logout", use_container_width=True):
            logout()
