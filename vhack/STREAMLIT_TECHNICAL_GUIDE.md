# Streamlit Technical Implementation Guide
## Digital Machinery Caretaker — Predictive Maintenance Platform

**Date:** March 12, 2026  
**Platform:** Streamlit + Gemini API  
**Target Users:** Factory Managers, Maintenance Technicians  

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Setup & Installation](#setup--installation)
3. [Project Structure](#project-structure)
4. [Page Specifications](#page-specifications)
5. [Gemini API Integration](#gemini-api-integration)
6. [Database Schema](#database-schema)
7. [API Integration with ML Backend](#api-integration-with-ml-backend)
8. [Component Details](#component-details)
9. [Authentication & Security](#authentication--security)
10. [Deployment](#deployment)

---

## Project Architecture

### High-Level Flow

\`\`\`
User Input (Dashboard/Forms)
    ↓
Streamlit Frontend Pages
    ↓
    ├─→ ML Backend (REST API) [Teammate handles]
    ├─→ Gemini API (LLM reasoning)
    └─→ Database (PostgreSQL/Firestore)
    ↓
Display Results + Recommendations
    ↓
Action (Approve, Export, Schedule Maintenance)
\`\`\`

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.28+ | Interactive dashboard & UI |
| **LLM** | Google Gemini 2.0 Flash | Root cause reasoning, decisions, explanations |
| **Visualization** | Plotly, Pandas | Charts, tables, Gantt diagrams |
| **Backend** | REST API (from ML team) | RUL predictions, anomaly detection, health scores |
| **Database** | PostgreSQL / Firestore | Machine data, repair history, financial parameters |
| **Document Processing** | Gemini Vision (multimodal) | Parse uploaded PDFs, extract financial data |
| **Deployment** | Docker / Streamlit Cloud | Production hosting |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip or poetry for dependency management
- Google Cloud account with Gemini API access
- PostgreSQL 12+ or Firestore access
- Access to ML backend API endpoint (from teammate)

### Installation Steps

#### 1. Clone/Create Project Repository

\`\`\`bash
mkdir machinery-caretaker-streamlit
cd machinery-caretaker-streamlit
git init
\`\`\`

#### 2. Create Virtual Environment

\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

#### 3. Install Dependencies

Create \`requirements.txt\`:

\`\`\`txt
streamlit==1.28.1
google-generativeai==0.3.0
pandas==2.0.3
plotly==5.17.0
requests==2.31.0
python-dotenv==1.0.0
sqlalchemy==2.0.21
psycopg2-binary==2.9.9
Pillow==10.0.0
PyPDF2==4.0.1
pydantic==2.4.2
\`\`\`

Install:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

#### 4. Environment Configuration

Create \`.streamlit/secrets.toml\`:

\`\`\`toml
# Google Gemini API
GEMINI_API_KEY = "your_gemini_api_key_here"

# Database
DATABASE_URL = "postgresql://user:password@localhost:5432/machinery_caretaker"
# OR for Firestore:
FIREBASE_CREDENTIALS_JSON = "path/to/firebase-credentials.json"

# ML Backend API
ML_BACKEND_URL = "http://localhost:5000"  # Your teammate's API endpoint
ML_API_KEY = "your_ml_backend_api_key"

# Application Settings
APP_MODE = "production"  # or "development"
CACHE_TTL = 3600  # seconds
\`\`\`

#### 5. Verify Installation

\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

Expected output:
\`\`\`
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
\`\`\`

---

## Project Structure

\`\`\`
machinery-caretaker-streamlit/
│
├── streamlit_app.py                 # Main entry point
├── requirements.txt                 # Dependencies
├── .streamlit/
│   └── secrets.toml                # API keys (DO NOT COMMIT)
│   └── config.toml                 # Streamlit configuration
│
├── pages/
│   ├── 1_Dashboard.py              # Fleet overview
│   ├── 2_Machine_Analysis.py       # Machine details + root cause
│   ├── 3_Financial_Risk.py         # Cost calculator
│   ├── 4_Maintenance_Plans.py      # 3 decision options
│   └── 5_Schedule.py               # Gantt chart timeline
│
├── utils/
│   ├── __init__.py
│   ├── api_client.py               # REST API calls to ML backend
│   ├── gemini_client.py            # Gemini LLM integration
│   ├── database.py                 # PostgreSQL/Firestore operations
│   ├── document_parser.py          # PDF parsing & data extraction
│   ├── cache_manager.py            # Data caching logic
│   └── auth.py                     # Authentication helpers
│
├── config/
│   ├── __init__.py
│   ├── settings.py                 # Application settings
│   └── prompts.py                  # Gemini prompt templates
│
├── models/
│   ├── __init__.py
│   ├── machine.py                  # Machine data model
│   ├── prediction.py               # ML prediction model
│   └── financial.py                # Financial analysis model
│
├── tests/
│   ├── test_api_client.py
│   ├── test_gemini_client.py
│   └── test_database.py
│
└── README.md                        # Documentation
\`\`\`

---

## Page Specifications

### Page 1: Dashboard (pages/1_Dashboard.py)

**Purpose**: Fleet overview of all machines; quick health status; critical alerts.

**Key Components**:
1. Sidebar Navigation
2. Key Metrics Row (Healthy, Warning, Critical, Avg RUL)
3. Machine Status Table (interactive dataframe)
4. Critical Alerts Section (containers with action buttons)
5. RUL Trend Chart (Plotly area chart for next 14 days)

**Functionality**:
- Display real-time machine health metrics
- Color-coded status indicators (🟢 🟡 🔴)
- Quick navigation to detailed machine analysis
- Forecast future fleet degradation
- Alert filtering and sorting

---

### Page 2: Machine Analysis (pages/2_Machine_Analysis.py)

**Purpose**: Deep-dive into single machine; sensor trends; root cause explanation via Gemini.

**Key Components**:
1. Machine Selector & Metrics Display
2. Sensor Trends Visualization (3 tabs: Vibration, Temperature, Frequency)
3. Root Cause Analysis Section (via Gemini LLM)
   - Detailed technical analysis
   - Executive summary (non-technical)
   - Technician guidance
4. Degradation Timeline (historical event markers)

**Functionality**:
- Interactive sensor charts (last 30 days)
- FFT frequency analysis display
- Gemini-powered explanations for different audiences
- Timeline showing when degradation started
- Evidence-based reasoning for failure prediction

---

### Page 3: Financial Risk (pages/3_Financial_Risk.py)

**Purpose**: Calculate downtime cost; upload/form-based input; cite document sources.

**Key Components**:
1. Input Method Toggle (Upload Document vs. Fill Form)
2. Document Upload Section
   - Parse PDF/TXT/DOCX using Gemini Vision
   - Extract financial parameters (hourly value, MTTR, costs)
   - Show extracted data with confidence levels
3. Form Input Section (text inputs for all parameters)
4. Cost Calculation Engine
   - Production loss = hourly_value × MTTR
   - Total downtime cost = production_loss + repair_cost + SLA + penalties
5. Results Display
   - Cost breakdown table with citations
   - Source attribution for each value
   - Preventative vs. failure cost comparison
6. Gemini Financial Recommendation
   - Should maintenance be done now or delayed?
   - Financial ROI justification
   - Management-level recommendation

**Functionality**:
- Parse uploaded documents for structured data
- Cite which document each value came from
- Interactive calculator (change inputs, see results update)
- Export report as PDF/Markdown/CSV
- Actionable financial recommendations powered by LLM

---

### Page 4: Maintenance Plans (pages/4_Maintenance_Plans.py)

**Purpose**: Generate 3 maintenance strategies via Gemini; show pros/cons; allow approval.

**Key Components**:
1. Machine Selector
2. 3 Plan Tabs (URGENT, BALANCED, DEFERRED)
   - For each plan:
     - Plan metrics (timeline, cost, risk level, success rate)
     - Maintenance actions (step-by-step list)
     - Resource requirements table
     - Risk assessment
     - Contingency plan (expandable)
     - When to use this plan (info box)
3. Decision Section
   - Approve button → logs approval, generates work order
   - Email button → sends plan to maintenance team
   - Download button → exports as PDF

**Functionality**:
- Gemini generates 3 distinct plans based on RUL, cost, resources
- Each plan uses different tradeoffs (speed vs cost, safety vs resources)
- Structured plan selection UI
- Approval workflow with notifications
- History tracking of selected plans

---

### Page 5: Schedule (pages/5_Schedule.py)

**Purpose**: Show upcoming maintenance as Gantt chart; resource allocation view.

**Key Components**:
1. Gantt Chart Visualization (next 30 days)
   - X-axis: Timeline
   - Y-axis: Machines
   - Colors: Status (Pending, Scheduled, Urgent)
   - Hoverable task info
2. Detailed Schedule Table
   - Machine ID, Status, Recommended Date, Duration, Assigned Team, Status
   - Sortable/filterable
3. Resource Utilization Chart
   - Team name → scheduled hours / available hours
   - Utilization percentage

**Functionality**:
- Visual timeline for maintenance planning
- Resource conflict detection
- Team utilization forecasting
- Drag-to-reschedule (optional, advanced feature)

---

## Gemini API Integration

### Prompt Templates (config/prompts.py)

Three core prompts:

**ROOT_CAUSE_PROMPT**:
- Input: ML data (RUL, vibration, temperature, anomalies, frequency peaks, confidence)
- Output: Root cause diagnosis + evidence + timeline + inspection points + risk assessment
- Format: Detailed (technical), Executive (non-technical), Technical (step-by-step for technicians)

**FINANCIAL_PROMPT**:
- Input: Machine RUL, failure cost, preventative cost, MTTR, savings
- Output: Should maintenance be done now or delayed? + financial justification + risk implications
- Audience: Management decision-makers

**MAINTENANCE_PLANS_PROMPT**:
- Input: Machine ID, RUL, diagnosed issue, financial impact, available resources
- Output: 3 plans (URGENT, BALANCED, DEFERRED) with actions, timeline, cost, resources, success probability
- Each plan includes: Pros/cons, when to use, contingency

### LLM Client Module (utils/gemini_client.py)

**Class: GeminiClient**
- `__init__(api_key)`: Initialize Gemini API
- `generate_root_cause(machine_id, ml_context)`: Root cause analysis
- `generate_financial_recommendation(machine_id, context)`: Financial decision support
- `generate_maintenance_plans(machine_id, context)`: 3-plan generation
- Helper methods for parsing Gemini responses

**Caching**: Use `@st.cache_data(ttl=3600)` to avoid re-querying Gemini for same machine

---

## Database Schema

### Core Tables (PostgreSQL)

1. **machines**: Machine metadata (ID, type, location, status)
2. **ml_predictions**: RUL, health score, anomalies, confidence
3. **sensor_history**: Raw sensor data (vibration, temperature, load) over time
4. **financial_parameters**: Machine-specific costs (hourly value, MTTR, repair cost, SLA penalties)
5. **maintenance_recommendations**: Generated plans with approval status
6. **maintenance_schedules**: Planned maintenance dates, assigned teams
7. **repair_history**: Actual repairs performed (for feedback loop and model improvement)
8. **documents**: Uploaded financial documents with extracted data

### Key Indexes
- `idx_machine_status`: Quick filtering by status
- `idx_prediction_machine`: Latest prediction per machine
- `idx_schedule_date`: Timeline queries

---

## API Integration with ML Backend

### REST API Endpoints (from ML team)

Your ML backend should expose:

**GET /machines**
- Returns: List of all machines with current status

**GET /machines/{machine_id}/prediction**
- Returns: Latest RUL, health score, vibration_rms, temperature, anomalies, frequency_peaks, confidence

**GET /machines/{machine_id}/history?sensor_type=&days=**
- Returns: Historical sensor data for visualization

**GET /machines/{machine_id}**
- Returns: Machine metadata

### API Client (utils/api_client.py)

**Class: MLBackendClient**
- Constructor: takes base_url, api_key
- Methods: 
  - `fetch_all_machines()`
  - `fetch_ml_prediction(machine_id)`
  - `fetch_machine_history(machine_id, sensor_type, days)`
  - `fetch_machine_info(machine_id)`
- Error handling: retry with exponential backoff
- Caching: `@st.cache_data` with TTL

---

## Component Details

### Data Caching Strategy (utils/cache_manager.py)

**Cache TTLs**:
- Machine list: 1 hour (rarely changes)
- ML predictions: 5 minutes (fresh RUL needed frequently)
- Sensor history: 10 minutes (for charts)
- Financial calculations: 30 minutes (user-entered data)
- Gemini analysis: 1 hour (same machine, same analysis)

**Implementation**: Use `@st.cache_data(ttl=seconds)` decorator

### Document Parser (utils/document_parser.py)

**Class: DocumentParser**
- Uses Gemini Vision API to read images/PDF content
- Extracts structured financial data from documents
- Returns: Dictionary of extracted values + source citations + confidence levels

**Extraction targets**:
- Hourly production value
- Units per hour
- Unit price
- Repair costs
- MTTR
- SLA penalty terms
- Supply chain penalties

---

## Authentication & Security

### Simple Auth System (utils/auth.py)

**Roles**:
- Manager: Can see financial impact, approve maintenance, export reports
- Technician: Can see analysis, mark repairs as complete
- Admin: Full access

**Methods**:
- `require_login()`: Check if user logged in
- `check_role_permission(role)`: Verify user role
- `login_page()`: Display login form
- `logout()`: Clear session

**Secrets Management**:
- Never commit `.streamlit/secrets.toml`
- Use Streamlit Cloud secrets for production

---

## Deployment Options

### Option 1: Local Development
\`\`\`bash
streamlit run streamlit_app.py
# Runs on http://localhost:8501
\`\`\`

### Option 2: Docker
- Create Dockerfile with Python 3.11
- Install requirements
- Expose port 8501
- Run: \`docker run -p 8501:8501 machinery-caretaker\`

### Option 3: Streamlit Cloud
- Push to GitHub
- Go to share.streamlit.io
- Connect repo
- Add secrets in dashboard settings

---

## Key Implementation Notes

### Critical Success Factors

1. **Gemini Prompt Engineering**: Invest time in well-crafted prompts for consistent results
2. **Caching Strategy**: Balance fresh data with API cost/latency
3. **Error Handling**: Handle API failures gracefully; show user-friendly messages
4. **Document Parsing**: Test PDF extraction thoroughly; handle edge cases
5. **Session State**: Use `st.session_state` to pass data between pages (e.g., selected_machine, financial_calc)
6. **Performance**: Profile with large datasets; optimize queries

### Development Workflow

1. Start with Page 1 (Dashboard) to validate API integration
2. Build Page 2 (Analysis) to test Gemini integration
3. Add Page 3 (Financial Risk) for document parsing
4. Implement Page 4 (Plans) for LLM decision generation
5. Finish Page 5 (Schedule) for visualization
6. Test end-to-end workflow
7. Deploy and iterate based on feedback

---

## Summary Checklist

- [ ] Project repository created
- [ ] Virtual environment setup
- [ ] Requirements.txt prepared
- [ ] Gemini API key obtained and configured
- [ ] ML backend API documented (from teammate)
- [ ] Database schema implemented
- [ ] All 5 pages planned and spec'd
- [ ] Gemini prompts drafted
- [ ] API client modules spec'd
- [ ] Authentication logic planned
- [ ] Deployment strategy chosen
- [ ] Documentation completed

---

**Status**: Architecture and specifications complete. Ready for implementation phase.
