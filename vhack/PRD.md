# Product Requirements Document (PRD)
## Digital Machinery Caretaker: AI-Driven Predictive Maintenance System for SMEs

**Version:** 1.0  
**Date:** March 12, 2026  
**Status:** Approved for Implementation  
**Target Market:** ASEAN Manufacturing SMEs  

---

## Executive Summary

**Digital Machinery Caretaker** is an AI-powered predictive maintenance platform that enables ASEAN Small and Medium Enterprises (SMEs) to transition from reactive/preventative maintenance to **data-driven proactive maintenance**.

By combining machine learning (RUL prediction), LLM-powered reasoning (Gemini), and an intuitive Streamlit dashboard, the system helps factory managers and technicians predict failures, understand root causes, quantify financial risks, and make optimal maintenance decisions.

**Key Innovation:** Unlike traditional "predict-only" systems, our platform adds **explainability, financial impact quantification, and decision support** — solving the trust and adoption barriers that prevent SMEs from using predictive maintenance.

---

## Problem Statement

### Current State (Industry Reality)

**ASEAN Manufacturing SME Maintenance Practices:**
- **65% use reactive maintenance**: Fix after failure occurs
- **30% use preventative maintenance**: Replace parts on fixed schedules (too early)
- **5% use predictive maintenance**: Too expensive, requires AI expertise

### Pain Points

| Issue | Impact | Frequency |
|-------|--------|-----------|
| Unplanned downtime | $50K-$500K per failure event | 3-5x per year per facility |
| Excessive maintenance costs | 15-30% overspend on preventative replacement | Ongoing |
| No failure visibility | Reactive firefighting, supply chain disruption | Constant |
| Maintenance expertise gap | Technicians lack data-driven decision tools | Persistent |
| AI distrust | "Black box" predictions not trusted by operators | Major adoption barrier |

### Market Opportunity

- **TAM (Total Addressable Market)**: ~50,000 ASEAN manufacturing SMEs
- **Addressable Market** (with 5+ critical machines): 15,000 SMEs
- **Total Market Size**: $750M annually (in downtime costs that could be prevented)
- **Customer Pain Urgency**: CRITICAL — downtime directly impacts profitability

---

## Solution Overview

### System Name
**Digital Machinery Caretaker** — Your AI assistant for machine health and maintenance

### Core Value Proposition

> **"Predict failures before they happen. Understand why. Make better maintenance decisions. Save money."**

### What We Solve

| Problem | Solution | Benefit |
|---------|----------|---------|
| Blind to degradation | Continuous sensor monitoring + ML RUL prediction | Early warning (days not hours) |
| No explanation for predictions | Gemini LLM analyzes ML outputs + technical reasoning | Trust + operator buy-in |
| Hidden financial impact | Automatic downtime cost calculation with citations | ROI-driven approval process |
| Resource optimization is hard | 3 maintenance plans (URGENT/BALANCED/DEFERRED) | Better tradeoff decisions |
| Poor maintenance tracking | Closed-loop feedback (repair reports → model improvement) | Better predictions over time |

---

## Product Architecture

### Four-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: DATA & SENSOR PROCESSING                          │
├─────────────────────────────────────────────────────────────┤
│ • Ingest multivariate time-series (vibration, temp, load)   │
│ • Handle noise, outliers, missing data                      │
│ • Extract degradation indicators (RMS, trend, spectrum)     │
│ • Clean dataset → high-quality features                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: PREDICTIVE ML ENGINE                              │
├─────────────────────────────────────────────────────────────┤
│ • RUL Prediction: XGBoost + LSTM ensemble                   │
│ • Anomaly Detection: Multi-method (Isolation Forest, etc.)  │
│ • Change-Point Detection: CUSUM algorithm                   │
│ • Health Score: Normalized 0-100 scale                      │
│ • Confidence Intervals: Quantile-based uncertainty          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: AI REASONING & DECISION SUPPORT (LLM)            │
├─────────────────────────────────────────────────────────────┤
│ ✓ Root Cause Reasoning                                      │
│   - Analyzes ML outputs (vibration, temp, frequency peaks)  │
│   - Infers likely failure mode (bearing wear, imbalance...)│
│   - Provides evidence-based diagnosis                       │
│   - Outputs: Detailed, Executive, & Technical versions      │
│                                                              │
│ ✓ Financial Risk Estimation                                 │
│   - Calculates downtime cost from uploaded docs             │
│   - Production loss + repair cost + SLA penalties           │
│   - ROI analysis (prevent now vs. fail later)               │
│                                                              │
│ ✓ Maintenance Decision Support                              │
│   - Generates 3 distinct plans (URGENT/BALANCED/DEFERRED)   │
│   - Each plan: timeline, cost, resources, success prob.     │
│   - Tradeoff analysis: speed vs. cost vs. safety            │
│   - When-to-use guidance for each option                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4: OPERATIONAL DASHBOARD & MAINTENANCE PLATFORM      │
├─────────────────────────────────────────────────────────────┤
│ • 5 Streamlit Pages (Dashboard, Analysis, Finance, Plans, Schedule)
│ • Real-time machine health visualization                    │
│ • Work order generation & approval workflow                 │
│ • Team assignment & resource utilization tracking          │
│ • Feedback loop: Technician repair reports → model learns  │
│ • REST API for third-party SCADA/MES integration           │
└─────────────────────────────────────────────────────────────┘
```

---

## Product Features

### Feature 1: Real-Time Machine Health Dashboard
**What**: One-screen view of all machines' health status
- 🟢 Healthy (normal operation)
- 🟡 Warning (attention needed)
- 🔴 Critical (urgent maintenance required)

**Who**: Factory managers, shift supervisors  
**When**: Daily use, decision trigger  
**Why**: Quick situational awareness; enables proactive scheduling

**Metrics Displayed**:
- RUL countdown (hours/days until failure)
- Health score (0-100%)
- Key anomalies (vibration ↑42%, temperature +5°C)
- Forecast: expected fleet status in 14 days

---

### Feature 2: Machine Analysis with Root Cause Explanation
**What**: Deep-dive into specific machine with AI-powered diagnosis

**Panels**:
1. **Sensor Trends** (last 30 days with anomaly markers)
   - Vibration RMS history
   - Temperature progression
   - Frequency spectrum (FFT analysis)

2. **AI Root Cause Analysis** (via Gemini)
   - Technical explanation (for engineers)
   - Executive summary (non-technical, 2-3 sentences)
   - Technician guidance (step-by-step inspection checklist)

3. **Degradation Timeline** (historical markers)
   - When vibration spike started
   - When CUSUM algorithm flagged trend change
   - When temperature increase became significant
   - Narrative of failure progression

**Why This Matters**: Operators TRUST predictions when they understand the reasoning. Black-box models are rejected; explainability drives adoption.

---

### Feature 3: Financial Risk Calculator
**What**: Quantify downtime cost to justify maintenance decisions

**Input Methods**:
- **Upload Mode**: Parse SLA/production spec PDF → extract financial parameters
- **Form Mode**: Manually enter hourly value, MTTR, repair costs, SLA penalties

**Output**:
- Cost breakdown table with source citations (which document each value came from)
- Total downtime cost if machine fails
- Preventative maintenance cost
- Net savings if acted today
- ROI timeline (when maintenance pays for itself)

**Example**: "If Pump_03 fails: $87,500 loss. Maintenance now: $6,000 cost. Savings: $81,500 (93% reduction). **Decision: Approve maintenance today.**"

---

### Feature 4: 3-Plan Maintenance Decision Support
**What**: AI generates 3 distinct maintenance strategies with different tradeoffs

**Plan 1: URGENT** (🚨 Safety-First)
- Timeline: 2-4 hours
- Cost: Higher (premium parts/labor)
- Risk: Very low (immediate action)
- Best for: Critical machines, zero-tolerance downtime
- Success rate: 95%

**Plan 2: BALANCED** (⚖️ Optimized)
- Timeline: 24-48 hours
- Cost: Standard
- Risk: Low-medium (allows resource planning)
- Best for: Normal operations, budget-conscious
- Success rate: 88%

**Plan 3: DEFERRED** (📅 Long-term Planning)
- Timeline: 5-7 days
- Cost: Lower (standard parts, planned labor)
- Risk: Medium-high (accept elevated risk short-term)
- Best for: When budget/resources are constrained
- Success rate: 72%

Each plan includes: Actions, resources, cost estimate, contingency plan, when-to-use guidance

---

### Feature 5: Maintenance Schedule & Resource Allocation
**What**: Visual timeline of all planned maintenance; team utilization view

**Visualizations**:
- Gantt chart (next 30 days)
- Resource utilization dashboard (team hours scheduled vs. available)
- Conflict detection (overlapping high-priority jobs)

