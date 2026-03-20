"""
Gemini API client with graceful mock fallback.
If no GEMINI_API_KEY is configured, realistic pre-written responses are returned
so the full UX is functional without an API key (e.g., for demo / hackathon).
"""

import json
import streamlit as st
from typing import Optional, Dict, Any

from config.settings import GEMINI_MODEL, CACHE_GEMINI_ANALYSIS
from config.prompts import ROOT_CAUSE_PROMPT, FINANCIAL_PROMPT, MAINTENANCE_PLANS_PROMPT

# ---------------------------------------------------------------------------
# Realistic mock responses keyed by machine status
# ---------------------------------------------------------------------------

_MOCK_ROOT_CAUSE = {
    "Critical": """\
## ROOT CAUSE ANALYSIS

### Primary Diagnosis
**Advanced gear tooth pitting on high-speed pinion shaft** — Probability: 93%

### Evidence Summary
- Vibration RMS 4.21 mm/s — 2.8× above critical threshold (1.5 mm/s); accelerating 0.15 mm/s per day
- Bearing housing temperature 89°C — 29°C above safe operating limit; rising at 0.8°C/day
- Gear mesh frequency (GMF) at 187 Hz with 3 harmonics and sideband spacing matching shaft rotation: classic gear tooth damage signature
- Metallic particle count in oil 4× ISO 4406 acceptable limit — confirming active metal-to-metal contact
- Anomaly sustained for 6+ consecutive days: entered irreversible rapid degradation zone

### Technical Analysis
The vibration spectrum reveals a clear gear defect pattern:
1. **GMF 1× at 187 Hz**: Amplitude 3.21 mm/s — 6.4× baseline. Indicates uneven load distribution across gear teeth, consistent with pitting or spalling on ≥3 teeth.
2. **GMF 2× & 3× harmonics**: Amplitudes 1.84 and 1.09 mm/s respectively. Presence of multiple harmonics confirms the defect is not isolated but affects a sector of the gear.
3. **Sidebands at GMF ± shaft frequency (12.3 Hz spacing)**: Asymmetric sideband amplitudes (0.98 vs. 0.87 mm/s) suggest eccentric loading — likely caused by worn input shaft bearing allowing shaft wobble.
4. **Temperature correlation**: Oil viscosity degrades exponentially above 80°C. At 89°C with Fe contamination, hydrodynamic lubrication film has collapsed, accelerating metal-to-metal contact and gear tooth fatigue.

Failure mode progression: micro-pitting → macro-pitting → spalling → tooth fracture. Based on current degradation rate, tooth fracture is expected within 4–7 days.

### Executive Summary
The high-speed gearbox is experiencing severe gear tooth damage with 93% confidence. The machine is in the rapid failure zone — without immediate intervention, a catastrophic gear fracture is expected within 5 days, risking unplanned production stoppage worth $45,000–$120,000 depending on the production batch in progress. Emergency maintenance authorization is required today.

### Technician Guidance
1. **STOP MACHINE** immediately if vibration exceeds 6.0 mm/s — catastrophic seizure risk
2. Lock-out/tag-out (LOTO) procedure on gearbox input shaft — verify zero energy state
3. Drain gearbox oil into a clear container — inspect for metallic shavings and discolouration
4. Remove gearbox inspection cover — visually examine high-speed pinion teeth for pitting, spalling, or cracks
5. Measure input shaft radial bearing clearance (acceptable: 0.02–0.05 mm; replace if >0.08 mm)
6. Check gear backlash on high-speed pair (acceptable: 0.15–0.25 mm; if >0.35 mm: gear set replacement required)
7. If oil is milky or has burnt odour: seal failure — replace seals before refill
8. After repair: run vibration baseline at 30%, 60%, 100% load for 30 min each; acceptable: <1.5 mm/s at all loads

### Risk Assessment
- Failure probability in 7 days: **78%**
- Failure probability in 14 days: **96%**
- Failure probability in 30 days: **99%**
- Consequence if unaddressed: Gear tooth fracture → secondary bearing damage → shaft seizure → 18–48 hr unplanned downtime

### Recommended Immediate Actions
1. Initiate emergency work order — target maintenance start within 24 hours
2. Pre-order replacement parts: high-speed pinion gear (GBX-HSP-001), 2× SKF 6308-2RS bearings, Viton shaft seals
3. Alert production scheduling to plan an 8-hour maintenance window — coordinate with shift supervisor
""",
    "Warning": """\
## ROOT CAUSE ANALYSIS

### Primary Diagnosis
**Progressive impeller wear with drive-end bearing degradation** — Probability: 72%

### Evidence Summary
- Vibration RMS 1.92 mm/s — 28% above warning threshold; trending upward at 0.04 mm/s per day
- Temperature +5°C above 30-day average; gradual rise consistent with increased bearing friction
- 1× shaft harmonic amplitude elevated 22% vs. baseline — indicative of mass imbalance or eccentricity
- No catastrophic failure indicators present; degradation is in early-to-mid stage

### Technical Analysis
The vibration signature suggests two co-developing issues:
1. **Impeller wear**: Elevated 1× and 2× shaft harmonics with slight amplitude modulation indicate asymmetric mass loss on impeller vanes — classic wear pattern for pumps handling mildly abrasive fluid.
2. **Drive-end bearing**: Sub-harmonic content and slight temperature rise at drive-end housing (63°C) suggest the bearing is entering the early warning zone. Lubrication re-greasing within 2 weeks could arrest progression.

Current degradation rate gives 38 days of estimated remaining useful life under nominal operating conditions. Operating at reduced capacity may extend this.

### Executive Summary
The pump is in a Warning state with two co-developing issues: impeller wear reducing efficiency and early bearing degradation. The machine is safe to operate for approximately 38 more days under normal conditions, but scheduling maintenance within the next 2 weeks is recommended to prevent progression to a critical failure scenario.

### Technician Guidance
1. Check pump discharge pressure and flow rate against design specification — a >10% efficiency drop confirms impeller wear
2. Measure drive-end bearing housing temperature with contact thermometer — compare left/right axial positions
3. Check coupling alignment (acceptable: <0.05 mm angular, <0.08 mm parallel offset)
4. Inspect mechanical seal for leakage — even a small drip indicates seal degradation accelerating bearing contamination
5. Re-grease drive-end bearing if last lubrication >3 months ago (use SKF LGEP 2 grease, 15g per bearing)
6. Listen for any high-pitched squealing — indicates bearing running dry

### Risk Assessment
- Failure probability in 7 days: **12%**
- Failure probability in 14 days: **28%**
- Failure probability in 30 days: **67%**
- Consequence if unaddressed: Bearing seizure or impeller fracture → 8–16 hr repair downtime

### Recommended Immediate Actions
1. Schedule preventive maintenance within 14 days (before RUL reaches 24 days)
2. Re-grease drive-end bearing this week as a low-cost risk mitigation ($15 cost vs. $38,000 failure cost)
3. Log vibration trend daily — escalate to URGENT if RMS exceeds 2.5 mm/s
""",
    "Healthy": """\
## ROOT CAUSE ANALYSIS

### Primary Diagnosis
**No failure mode detected** — Machine operating within all normal parameters

### Evidence Summary
- Vibration RMS 0.82 mm/s — well within healthy range (0.5–1.5 mm/s); stable trend
- Temperature 47.3°C — 12.7°C below warning threshold; no thermal anomalies
- Anomaly score 0.05 — effectively zero; model confidence 94%
- No abnormal frequency components detected in spectrum

### Technical Analysis
All sensor readings are within the expected normal operating envelope for this machine type:
- **Vibration**: Only 1× and 2× shaft harmonics present at low amplitudes — consistent with balanced rotating assembly
- **Temperature**: Stable with normal daily operational cycle — no evidence of lubrication failure, overloading, or electrical fault
- **Frequency spectrum**: Clean spectrum with no gear mesh harmonics, bearing defect frequencies, or modulation sidebands

The ML model assigns 94% confidence to the "no failure" classification, supported by 145 days of remaining useful life projection.

### Executive Summary
This machine is in excellent health with no maintenance intervention required at this time. All monitored parameters are nominal, and the predictive model projects continued reliable operation for approximately 145 days. A routine preventive maintenance check can be scheduled as part of planned downtime rather than as a priority action.

### Technician Guidance
1. Continue monitoring vibration and temperature readings — no action required if trends remain stable
2. Perform routine lubrication at next scheduled service interval
3. Check coupling alignment and foundation bolts at next planned shutdown
4. No immediate inspection items required

### Risk Assessment
- Failure probability in 7 days: **<1%**
- Failure probability in 14 days: **<2%**
- Failure probability in 30 days: **4%**
- Consequence if unaddressed: Gradual normal wear — no near-term risk

### Recommended Immediate Actions
1. No immediate action required — schedule routine preventive maintenance at 120-day mark
2. Continue monitoring per normal schedule (daily vibration + temperature readings)
3. Update maintenance log with current health assessment
""",
}

_MOCK_FINANCIAL = {
    "approve": """\
## FINANCIAL RECOMMENDATION

### Decision
**APPROVE MAINTENANCE IMMEDIATELY**

### Financial Justification
The preventive maintenance investment of $8,500 protects against a total failure cost of $127,300 — a 1,397% return on maintenance investment. With RUL at only 5 days, every 24-hour delay increases the probability of catastrophic failure by approximately 11 percentage points, adding ~$14,000 in expected loss per day of inaction.

### Risk-Adjusted Analysis
- **Act today**: Spend $8,500. Guaranteed savings of $118,800. Production resumes in 8 hours.
- **Wait 1 week**: 78% probability of catastrophic failure. Expected cost: 0.78 × $127,300 + 0.22 × $8,500 = $101,190. Expected loss vs. acting today: $92,690.
- **Wait 2 weeks**: Near-certain failure (96%). Expected cost: $124,428. Net loss of ~$116,000 vs. acting today.

### Management Recommendation
Approve the $8,500 emergency maintenance work order immediately. The financial case is unambiguous: the ROI is 1,397% and the cost of inaction compounds daily. This machine should be taken offline for planned repair within 24 hours before the failure becomes uncontrolled.

### Key Metrics Summary
- Investment:    $8,500
- Return:        $118,800
- ROI:           1,397%
- Payback:       Immediate
""",
    "monitor": """\
## FINANCIAL RECOMMENDATION

### Decision
**SCHEDULE WITHIN 7 DAYS**

### Financial Justification
The preventive maintenance cost of $4,200 averts a projected failure cost of $38,500 — a 817% ROI. The current RUL of 38 days provides a planning window, but delaying beyond 14 days increases the expected cost by approximately $3,200 per week as failure probability rises.

### Risk-Adjusted Analysis
- **Act within 7 days**: Spend $4,200. High confidence of averting failure. Net savings: $34,300.
- **Wait 2 weeks**: Failure probability rises to 28%. Expected cost: 0.28 × $38,500 + 0.72 × $4,200 = $13,802. Expected additional loss: $9,600.
- **Wait 1 month**: Failure probability: 67%. Expected cost: $28,595. Net loss of ~$24,400 vs. acting now.

### Management Recommendation
Schedule this machine for preventive maintenance within the next 7 days during a planned production break. The 817% ROI justifies immediate scheduling, and the 38-day RUL provides flexibility for proper planning without emergency-rate labor costs.

### Key Metrics Summary
- Investment:    $4,200
- Return:        $34,300
- ROI:           817%
- Payback:       Immediate
""",
}

_MOCK_PLANS = {
    "plans": [
        {
            "plan_type": "URGENT",
            "headline": "Emergency overhaul — take offline within 24 hours",
            "timeline_hours": 8,
            "estimated_cost_usd": 8500,
            "risk_level": "Very Low",
            "success_rate_pct": 95,
            "best_for": "Critical machines where continued operation poses catastrophic failure risk within days",
            "actions": [
                "Coordinate emergency production stop with shift supervisor (30 min)",
                "Apply Lock-Out/Tag-Out (LOTO) on all energy sources",
                "Drain and safely dispose of contaminated gearbox oil (4.5L ISO VG 220)",
                "Remove gearbox endcap — photograph gear condition before disassembly",
                "Replace high-speed pinion gear (Part #GBX-HSP-001)",
                "Replace input shaft bearings: 2× SKF 6308-2RS",
                "Replace Viton shaft seals — both input and output sides",
                "Flush and clean gearbox housing — remove all metallic debris",
                "Refill with fresh ISO VG 220 synthetic gear oil",
                "Recouple drive motor — perform laser alignment (target: <0.03 mm)",
                "Commission test: 30 min at 30%, 60%, 100% load",
                "Verify: vibration <1.5 mm/s and temperature <65°C before release",
            ],
            "parts_required": [
                {"name": "High-Speed Pinion Gear", "part_number": "GBX-HSP-001", "quantity": 1, "unit_cost": 3200},
                {"name": "SKF Deep Groove Bearing 6308-2RS", "part_number": "SKF-6308-2RS", "quantity": 2, "unit_cost": 180},
                {"name": "Viton Shaft Seal Kit", "part_number": "SEAL-VSK-42", "quantity": 1, "unit_cost": 95},
                {"name": "ISO VG 220 Synthetic Gear Oil (5L)", "part_number": "OIL-VG220-5L", "quantity": 1, "unit_cost": 75},
            ],
            "labor_requirements": [
                {"role": "Senior Maintenance Technician", "hours": 6},
                {"role": "Mechanical Technician (assistant)", "hours": 8},
            ],
            "pros": [
                "Eliminates near-certain catastrophic failure within 5 days",
                "Restores machine to full operational health",
                "Highest success rate (95%)",
                "Prevents secondary damage (bearings, housing) from delayed action",
            ],
            "cons": [
                "Requires immediate unplanned production stop (8 hours)",
                "Premium parts may need expedited shipping",
                "Higher cost than deferred approach",
            ],
            "contingency_plan": "If pinion gear part unavailable: install temporary 20% load-reduced operating mode and expedite parts within 48 hrs. If secondary shaft damage found during inspection: escalate to full gearbox replacement (Part #GBX-ASSY-7700, 16-hr job).",
        },
        {
            "plan_type": "BALANCED",
            "headline": "Planned overhaul within 48 hours — optimised for cost and resources",
            "timeline_hours": 36,
            "estimated_cost_usd": 6800,
            "risk_level": "Low",
            "success_rate_pct": 88,
            "best_for": "When a short planning window is available to procure parts and assign the optimal team",
            "actions": [
                "Order replacement parts today (same-day delivery available for critical items)",
                "Schedule 8-hour maintenance window in next 36–48 hours with production planning",
                "Assign Senior Technician + 1 assistant for the repair window",
                "Apply LOTO procedure at scheduled window start",
                "Perform full gearbox inspection: gear set, bearings, seals, shaft",
                "Replace pinion gear and bearings as planned",
                "Perform shaft alignment and oil service",
                "Commission test per standard procedure",
                "Submit repair report and update CMMS",
            ],
            "parts_required": [
                {"name": "High-Speed Pinion Gear", "part_number": "GBX-HSP-001", "quantity": 1, "unit_cost": 3200},
                {"name": "SKF Deep Groove Bearing 6308-2RS", "part_number": "SKF-6308-2RS", "quantity": 2, "unit_cost": 180},
                {"name": "Viton Shaft Seal Kit", "part_number": "SEAL-VSK-42", "quantity": 1, "unit_cost": 95},
                {"name": "ISO VG 220 Synthetic Gear Oil (5L)", "part_number": "OIL-VG220-5L", "quantity": 1, "unit_cost": 75},
            ],
            "labor_requirements": [
                {"role": "Senior Maintenance Technician", "hours": 6},
                {"role": "Mechanical Technician (assistant)", "hours": 8},
            ],
            "pros": [
                "Allows proper resource planning and parts procurement",
                "Avoids emergency labor rate premium (~15% cost saving vs. URGENT)",
                "Minimal production disruption with pre-planned downtime window",
                "Lower cost than URGENT plan ($1,700 savings)",
            ],
            "cons": [
                "36–48 hour window carries 18% additional failure risk vs. URGENT",
                "Requires continuous monitoring during planning period",
                "If vibration exceeds 6.0 mm/s before maintenance window: must escalate to URGENT",
            ],
            "contingency_plan": "Install continuous vibration monitor alert at 5.0 mm/s threshold. If triggered before planned window, escalate to URGENT plan immediately. Assign on-call technician for rapid response.",
        },
        {
            "plan_type": "DEFERRED",
            "headline": "Monitored deferral — 5 to 7 days with enhanced monitoring",
            "timeline_hours": 144,
            "estimated_cost_usd": 5200,
            "risk_level": "High",
            "success_rate_pct": 62,
            "best_for": "Only when budget or resource constraints make earlier action impossible — requires continuous monitoring commitment",
            "actions": [
                "Install temporary vibration monitor with 4.0 mm/s alarm threshold",
                "Reduce machine operating load by 30% to slow degradation rate",
                "Perform daily oil inspection — check for increased metallic particles",
                "Order all replacement parts for delivery within 5 days",
                "Schedule maintenance window for day 6–7",
                "Brief operations team on emergency stop criteria",
                "Perform maintenance as planned on day 6–7",
            ],
            "parts_required": [
                {"name": "High-Speed Pinion Gear", "part_number": "GBX-HSP-001", "quantity": 1, "unit_cost": 3200},
                {"name": "SKF Deep Groove Bearing 6308-2RS", "part_number": "SKF-6308-2RS", "quantity": 2, "unit_cost": 180},
                {"name": "Viton Shaft Seal Kit", "part_number": "SEAL-VSK-42", "quantity": 1, "unit_cost": 95},
                {"name": "Temporary Vibration Monitor (rental)", "part_number": "VIB-MON-RENTAL", "quantity": 1, "unit_cost": 150},
            ],
            "labor_requirements": [
                {"role": "Senior Maintenance Technician", "hours": 6},
                {"role": "Mechanical Technician", "hours": 8},
                {"role": "Monitoring Operator (daily checks)", "hours": 7},
            ],
            "pros": [
                "Allows standard-rate parts and labor procurement",
                "Minimal immediate production disruption",
                "Lowest upfront cost",
            ],
            "cons": [
                "62% success rate — 38% risk of uncontrolled failure before maintenance window",
                "If failure occurs unexpectedly: $127,300 total cost vs. $5,200 planned",
                "Secondary damage risk: housing, shaft, coupling may require replacement if catastrophic failure",
                "Reduced capacity operation impacts production throughput by ~30%",
                "Requires dedicated daily monitoring effort",
            ],
            "contingency_plan": "Define hard stop criterion: vibration >5.5 mm/s OR temperature >95°C → IMMEDIATE SHUTDOWN regardless of production schedule. Pre-position all parts and tools on-site from day 3 onwards for rapid response. Contact backup technician team to be on standby from day 4.",
        },
    ]
}


def _get_api_key() -> Optional[str]:
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


class GeminiClient:
    def __init__(self):
        self.api_key = _get_api_key()
        self._model = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(GEMINI_MODEL)
            except Exception:
                self._model = None

    def _call(self, prompt: str) -> str:
        if self._model is None:
            return None
        try:
            response = self._model.generate_content(prompt)
            return response.text
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_root_cause(self, machine_id: str, prediction, machine) -> str:
        """Returns root cause analysis markdown string."""
        anomalies_str = "\n".join(f"  • {a}" for a in prediction.anomalies) if prediction.anomalies else "  • None detected"
        peaks_str = "\n".join(
            f"  • {fp.frequency_hz} Hz — Amplitude {fp.amplitude:.2f} — {fp.label}"
            for fp in prediction.frequency_peaks
        )

        prompt = ROOT_CAUSE_PROMPT.format(
            machine_id=machine_id,
            machine_type=machine.machine_type,
            location=machine.location,
            status=machine.status,
            rul_days=prediction.rul_days,
            health_score=prediction.health_score,
            anomaly_score=prediction.anomaly_score,
            confidence=prediction.confidence,
            vibration_rms=prediction.vibration_rms,
            temperature_celsius=prediction.temperature_celsius,
            anomalies=anomalies_str,
            frequency_peaks=peaks_str,
        )

        result = self._call(prompt)
        if result:
            return result

        # Fallback to pre-written mock response
        return _MOCK_ROOT_CAUSE.get(machine.status, _MOCK_ROOT_CAUSE["Healthy"])

    def generate_financial_recommendation(self, machine_id: str, params, calc) -> str:
        """Returns financial recommendation markdown string."""
        prompt = FINANCIAL_PROMPT.format(
            machine_id=machine_id,
            machine_type="Industrial Machine",
            rul_days=calc.cost_breakdown.get("rul_days", 30),
            diagnosed_issue=params.source or "Detected degradation",
            hourly_production_value=params.hourly_production_value,
            mttr_hours=params.mttr_hours,
            preventative_cost=calc.total_preventative_cost,
            failure_repair_cost=params.repair_cost_failure,
            sla_penalty_per_hour=params.sla_penalty_per_hour,
            supply_chain_penalty=params.supply_chain_penalty,
            production_loss=calc.production_loss,
            total_failure_cost=calc.total_downtime_cost_failure,
            net_savings=calc.net_savings,
            roi_pct=calc.roi_percentage,
        )

        result = self._call(prompt)
        if result:
            return result

        key = "approve" if calc.roi_percentage > 300 else "monitor"
        return _MOCK_FINANCIAL[key]

    def generate_maintenance_plans(self, machine_id: str, machine, prediction, financial_impact: float) -> dict:
        """Returns parsed maintenance plans dict."""
        prompt = MAINTENANCE_PLANS_PROMPT.format(
            machine_id=machine_id,
            machine_type=machine.machine_type,
            location=machine.location,
            rul_days=prediction.rul_days,
            health_score=prediction.health_score,
            diagnosed_issue=prediction.predicted_failure_mode,
            financial_impact=financial_impact,
            available_teams="Team Alpha (gearboxes/heavy), Team Beta (pumps/fluid), Team Gamma (conveyors/motors)",
        )

        result = self._call(prompt)
        if result:
            try:
                # Strip markdown code fences if present
                clean = result.strip()
                if clean.startswith("```"):
                    clean = "\n".join(clean.split("\n")[1:])
                if clean.endswith("```"):
                    clean = "\n".join(clean.split("\n")[:-1])
                return json.loads(clean)
            except (json.JSONDecodeError, ValueError):
                pass

        return _MOCK_PLANS

    def generate_rul_drop_explanation_from_facts(
        self,
        machine_id: str,
        machine,
        prediction,
        structured_facts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate operator-friendly explanation using structured facts only."""
        facts_json = json.dumps(structured_facts, ensure_ascii=True)
        prompt = f"""
You are a maintenance advisor for non-technical factory operators.

Rules:
- Use only the structured facts below.
- Do not invent extra measurements.
- Keep language plain and actionable.
- Return strict JSON only.

Machine ID: {machine_id}
Machine Name: {machine.name}
Machine Status: {machine.status}
Current RUL Days: {prediction.rul_days:.1f}

Structured Facts:
{facts_json}

Output schema:
{{
  "summary": "2-4 short sentences on why RUL is decreasing",
  "actions": [
    {{
      "inspect_first": "component/system",
      "check_now": "immediate check",
      "plan_by": "deadline",
      "escalate_if": "condition"
    }}
  ],
  "evidence": ["bullet", "bullet"],
  "confidence": "low|medium|high"
}}
"""

        result = self._call(prompt)
        if result:
            try:
                clean = result.strip()
                if clean.startswith("```"):
                    clean = "\n".join(clean.split("\n")[1:])
                if clean.endswith("```"):
                    clean = "\n".join(clean.split("\n")[:-1])
                parsed = json.loads(clean)
                parsed.setdefault("summary", "No summary generated.")
                parsed.setdefault("actions", [])
                parsed.setdefault("evidence", [])
                parsed.setdefault("confidence", structured_facts.get("confidence", "medium"))
                return parsed
            except Exception:
                pass

        top = structured_facts.get("top_drivers", [])
        top_names = ", ".join([d.get("feature", "unknown") for d in top[:2]]) or "no dominant driver"
        risk = structured_facts.get("risk_band", machine.status)
        return {
            "summary": (
                f"RUL is trending down and the strongest contributors are {top_names}. "
                f"Current risk band is {risk}."
            ),
            "actions": [
                {
                    "inspect_first": top[0].get("feature", "vibration") if top else "vibration",
                    "check_now": "Compare current reading against warning threshold.",
                    "plan_by": "Within 24 hours" if risk in ["Critical", "Red"] else "Within 72 hours",
                    "escalate_if": "Trend accelerates or status becomes Critical.",
                }
            ],
            "evidence": structured_facts.get("anomalies", []),
            "confidence": structured_facts.get("confidence", "medium"),
        }
