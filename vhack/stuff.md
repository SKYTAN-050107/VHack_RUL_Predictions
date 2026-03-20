CASE STUDY 1:
Predictive Maintenance for SME Resilience

Track: Machine Learning (Time-Series / Remaining Useful Life Estimation)

Primary Goal: SDG 9: Industry, Innovation, and Infrastructure (Target 9.4)

1. Real-World Context

Small and Medium Enterprises (SMEs) are the backbone of the ASEAN
economy, yet they often operate with aging industrial machinery and thin
profit margins. Unlike large conglomerates, these factories cannot
afford "Smart Factory" overhauls. A single motor failure in a rural food
processing plant can halt production for weeks, leading to massive
financial losses and resource waste.

Predictive Maintenance for SME
Resilience
CASE STUDY 1:

Track: Machine Learning (Time-Series / Remaining Useful Life Estimation)

Primary Goal: SDG 9: Industry, Innovation, and Infrastructure
(Target 9.4)

2. Problem Statement

Current maintenance in ASEAN SMEs is largely reactive (fixing after
failure) or preventative (replacing parts too early). Both are inefficient.
There is a critical need for an AI-driven system that can analyze sensor
data,such as vibration, temperature, and load to predict the Remaining
Useful Life (RUL) of machinery, enabling proactive planning and reducing
downtime.

3. Technical Challenge & Sub-tasks

• Temporal Feature Engineering: Process high-frequency multivariate
time-series data to extract features that represent machine
degradation.
• RUL Regression Modeling: Develop a robust model to estimate the
precise number of cycles or hours remaining before a functional
failure.
• Anomaly Change-Point Detection: Implement logic to identify the
exact moment a machine transitions from a "Healthy" state to an
"Impaired" state.
• Health Dashboard Visualization: Design a user-friendly interface for
factory managers to visualize machine health and schedule
maintenance efficiently.

4. Technical Feasibility & Constraints

• Noisy Data Handling: Models must demonstrate the ability to handle
sensor noise and missing data points common in real-world industrial
settings.
• Model Interpretability: The AI must provide actionable insights (e.g.,
why the RUL is decreasing) to gain trust from non-technical factory
operators.
• Scalability: The solution should be modular, allowing it to be applied
to different types of machinery with minimal retraining.

5. Recommended Data Sources & Toolkits

• Suggested Datasets: NASA CMAPSS.
• Suggested Frameworks: Scikit-Learn, TensorFlow/PyTorch, Darts,
sktime, NeuralProphet, or Prophet.
• Visualization Tools: Streamlit or Dash.

6. Expected Deliverables

• Predictive Model: A validated model with clear performance metrics
(e.g., RMSE or MAE).
• Functional Dashboard: A prototype showing real-time machine
status and "Time-to-Failure" countdowns.

---

Proposed solution:

My teammate is doing ML modeling so, im doing product. 

Predictive Maintenance Workflow
1. Remaining Useful Life (RUL) Prediction
Action: ML model predicts remaining cycles/hours.

Input: Vibration, temperature, load sensors.

2. Convert RUL to Failure Window
Estimate number of days until breakdown.

Add prediction confidence interval.

3. Root Cause Reasoning
Explain why degradation happens.

Feature importance / anomaly signals.

Map signals to possible component issues.

4. Financial & Operational Risk Analysis
Estimate downtime cost.

Estimate repair cost.

Calculate total business impact.

5. Management Report
Failure window prediction.

Root cause explanation.

Financial risk estimate.

Recommended maintenance actions.

[Branching Point]

6A. Recommendation Planning
List possible repair solutions.

Prioritize lowest Mean Time To Repair (MTTR).

Select optimal maintenance strategy.

6B. Management Decision
Approve / delay / modify maintenance plan.

Choose option based on cost vs. downtime risk.

7. Generate Technical Instructions
Detailed maintenance steps.

Required components / repair type.

Estimated repair time.

8. Maintenance Platform
Active monitoring dashboard.

Maintenance scheduling.

Notification to technicians.

9. Technician Report
Inspection checklist.

Repair instructions.

Repair outcome logging.

10. Technician Feedback Loop
Actual failure cause.

Repair results.

Used for model improvement (Loops back to Step 1).

---

Tech stack:
Streamlit for dashboard and UI
FastAPI for backend server
Langchain for RAG
Supabase for db and vector db
Google Gemini API for LLM reasoning

---
UI:
Overall it should have just sidebar and main content.

Sidebar should have the page for:
- Overview
- Input resource like tech manual and financial report
- input machinery ML data
- maintenance platform

---
Overview should have columns of machines with names, RUL, and indicator in red, yellow, green color. If the machine is red, means it requires attention, the overview page should have a notifcation button at top right corner of the page.  clicking those will have details about the machine. normally it shoud have real time data update there. 

if the machine indicator is red, the notifiaction should pop up, the page details should show about root cause analysis, how bad can it be, and what to do with it.

root cause analysis is performed by LLM to see why ML model say its RUL is like that in normal language that a non-technical member can understand. 

for "how bad can it be" its like estimating downtime cost using "total downtime cost" formula, which the information will be taken from the files at input resource page

and for "what to do with it" its like giving a way on how to fix it, or replace what component, based on the file at input resource page. the "way" of fixing it shuld be 3, and it should prioritize either cost, time, or labour. these will be handled by RAG and LLM. 

---

input resource page should have 2 drag and drop box. 1 for technical resource, 1 for financial report. both will be used for giving grounded advice. and both files will be converted and put into vector db for RAG for LLM use. the page should have a check list under the title to make sure the target audience input all the required documents.

---

input ML data page is just drag and drop box for csv. i actually have no idea what to do with this part so i need assistance on this one. 

---
maintenance page should list all the ongoing maintenance plan, with technician name. when clicking it, it should show the detail, which will be root-cause analysis, and steps taken to do maintenance. after the technician completes the maintenance, the page should ask for filling maintenance form, and thei nfo will be updated for ML use, to update machine RUL.


