# Application configuration — single source of truth for constants and toggles.
import os

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# Set to False when the real ML backend is available.
USE_MOCK_DATA = True
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_TITLE = "Digital Machinery Caretaker"
APP_SUBTITLE = "AI-Driven Predictive Maintenance Platform"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Status thresholds
# ---------------------------------------------------------------------------
HEALTH_SCORE_THRESHOLDS = {
    "Healthy": (75, 100),
    "Warning": (50, 75),
    "Critical": (0, 50),
}

RUL_WARNING_DAYS = 60   # RUL below this → Warning
RUL_CRITICAL_DAYS = 20  # RUL below this → Critical

# ---------------------------------------------------------------------------
# UI Colours (aligned with PRD)
# ---------------------------------------------------------------------------
STATUS_COLORS = {
    "Healthy": "#2ECC71",
    "Warning": "#F39C12",
    "Critical": "#E74C3C",
}

STATUS_ICONS = {
    "Healthy": "🟢",
    "Warning": "🟡",
    "Critical": "🔴",
}

PLAN_COLORS = {
    "URGENT": "#E74C3C",
    "BALANCED": "#3498DB",
    "DEFERRED": "#9B59B6",
}

SCHEDULE_STATUS_COLORS = {
    "Urgent":    "#E74C3C",
    "Scheduled": "#3498DB",
    "Pending":   "#F39C12",
    "Completed": "#2ECC71",
}

# ---------------------------------------------------------------------------
# Cache TTLs (seconds)
# ---------------------------------------------------------------------------
CACHE_MACHINE_LIST = 3600
CACHE_ML_PREDICTIONS = 300
CACHE_SENSOR_HISTORY = 600
CACHE_GEMINI_ANALYSIS = 3600

# ---------------------------------------------------------------------------
# Gemini model
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.0-flash"
