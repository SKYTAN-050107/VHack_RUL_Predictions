import json
import re
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY
from services.rag_service import rag_service

class ReasoningService:
    def __init__(self):
        if not GOOGLE_API_KEY:
            print("Warning: GOOGLE_API_KEY not set for reasoning service.")
            self.llm = None
            return
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
        except:
            self.llm = None
            print("Warning: Failed to initialize Gemini Chat model.")

    async def generate_root_cause_analysis(
        self,
        machine_name: str,
        rul: int,
        context: List[Dict[str, Any]],
        shap_summary: str = "SHAP evidence unavailable.",
    ):
        """Generate root cause analysis and recommendations using RAG context and SHAP summary."""
        if not self.llm:
            return {
                "analysis": (
                    f"Root Cause Analysis for {machine_name} is currently unavailable (LLM not configured). "
                    f"Analysis based on RUL of {rul} cycles indicates potential Tier 2 component wear. "
                    f"SHAP evidence considered: {shap_summary}"
                ),
                "downtime_risk_cost": 7450.00,
                "criticality_level": "Semi-Critical (Level 2)",
                "downtime_cost_breakdown": "LLM OFFLINE: [($650 + $250) * 8 * 1.0] + $1,200 = $7,450.00",
                "discovery_path": "Fallback: Default industry values used from Q1 Financial Report and Maintenance SOP for Tier 2 failure on Semi-Critical machine.",
                "recommendations": [
                    {
                        "label": "Baseline Bearing Service", 
                        "priority": "Time",
                        "description": "Standard replacement of machine bearings.",
                        "rationale": "Recommended based on RUL thresholds for this machine type.",
                        "impact": "Extends RUL by 400 cycles.",
                        "cost": "$8,250",
                        "estimated_time": "8 hours",
                        "steps": "1. LOTO\n2. Disassemble\n3. Replace\n4. QA",
                        "components": "NSK-6205 Bearings"
                    },
                    {"label": "Lubrication & Calibration", "priority": "Cost", "description": "Minor service to stabilize machine.", "rationale": "Cost-effective short-term fix.", "impact": "Prevents immediate failure.", "cost": "$2,450", "estimated_time": "2 hours", "steps": "...", "components": "..."},
                    {"label": "Full System Audit", "priority": "Labor", "description": "Detailed inspection of all components.", "rationale": "Ensures long-term reliability.", "impact": "Identifies hidden faults.", "cost": "$12,000", "estimated_time": "12 hours", "steps": "...", "components": "..."}
                ]
            }

        context_text = "\n\n".join([c.get("content", "") for c in context])

        prompt = PromptTemplate.from_template("""
        You are a Senior Manufacturing Consultant and Reliability Engineer. Based on the provided Q1 Financial Risk Report and Maintenance SOP,
        conduct a Total Business Impact (TBI) assessment for the {machine_name} (Current RUL: {rul} cycles).

        ### MANDATORY CALCULATION:
        Use the Executive Standard formula from the Financial Report:
        TBI = [(Lost Sales Opportunity + Total Burn Rate) × MTTR × Criticality Multiplier] + Recovery Costs + Penalties

        ### DATA EXTRACTION:
        1. From FINANCIAL REPORT: Extract LSO, TBR (DLC + FO), Criticality Multipliers, and Penalties.
        2. From MAINTENANCE SOP: Determine the Failure Tier based on the machine's symptoms, MTTR benchmarks, and Recovery Tiers.
        3. Determine machine criticality (Bottleneck, Semi-Critical, or Support) based on its type and current role.

        CONTEXT FROM MANUALS & REPORTS:
        {context_text}

        MODEL EXPLAINABILITY EVIDENCE (SHAP TOP DRIVERS):
        {shap_summary}

        REQUIREMENT:
        In the "analysis" field, explicitly reference at least two SHAP features and explain
        how they contribute to degradation risk in plain language.

        Your response MUST be in JSON format with exactly six keys:
        1. "analysis": A high-level technical root cause analysis (2-3 sentences).
        2. "downtime_risk_cost": The calculated TBI as a float.
        3. "criticality_level": The determined criticality (e.g., "Bottleneck (1.5x)").
        4. "downtime_cost_breakdown": A concise string showing ONLY the numerical calculation.
           Example: "[(650 + 250) * 6.5 * 1.5] + 1200 = 9975.00"
        5. "discovery_path": A short paragraph (2-3 sentences) explaining HOW the AI found this information.
        6. "recommendations": A list of exactly 3 strategy objects with:
           "label", "priority", "description", "rationale", "impact", "cost", "estimated_time", "steps", "components".
        """)

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "machine_name": machine_name,
                    "rul": rul,
                    "context_text": context_text,
                    "shap_summary": shap_summary,
                }
            )

            content = str(response.content).strip()
            json_match = re.search(r"(\{.*\})", content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            parsed_data = json.loads(content)
            defaults = {
                "analysis": "Analysis completed.",
                "downtime_risk_cost": 0.0,
                "criticality_level": "Unknown",
                "downtime_cost_breakdown": "Calculation details extracted from report.",
                "discovery_path": "Data extracted from Financial Report and Maintenance SOP.",
                "recommendations": [],
            }
            for key, val in defaults.items():
                if key not in parsed_data:
                    parsed_data[key] = val

            return parsed_data
        except Exception as e:
            return {
                "analysis": f"Critical Error in Reasoning Service: {str(e)}",
                "downtime_risk_cost": 5000.00,
                "criticality_level": "Emergency Mode",
                "downtime_cost_breakdown": f"SYSTEM ERROR: {str(e)[:120]}",
                "discovery_path": f"AI was unable to complete discovery due to a system error: {str(e)[:120]}",
                "recommendations": [
                    {"label": "Manual Data Entry", "priority": "Time", "description": "Please check reports manually.", "rationale": "Service failure.", "impact": "N/A", "cost": "Unknown", "estimated_time": "Unknown", "steps": "N/A", "components": "N/A"},
                    {"label": "Review logs", "priority": "Cost", "description": "Check logs.", "rationale": "Error.", "impact": "N/A", "cost": "Unknown", "estimated_time": "Unknown", "steps": "N/A", "components": "N/A"},
                    {"label": "Contact Admin", "priority": "Labor", "description": "Contact admin.", "rationale": "Error.", "impact": "N/A", "cost": "Unknown", "estimated_time": "Unknown", "steps": "N/A", "components": "N/A"},
                ],
            }

    async def generate_rul_driver_explanation(
        self,
        machine_name: str,
        rul: int,
        structured_facts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate operator-facing explanation from curated structured facts only.
        Raw sensor tables/CSV are intentionally excluded from this path.
        """
        facts_json = json.dumps(structured_facts, ensure_ascii=True)
        trend = str(structured_facts.get("rul", {}).get("trend_direction", "stable")).lower()
        top = structured_facts.get("top_drivers", [])
        rul_now = float(structured_facts.get("rul", {}).get("rul_now", rul) or rul)
        rul_then = float(structured_facts.get("rul", {}).get("rul_then", rul_now) or rul_now)
        delta = float(structured_facts.get("rul", {}).get("rul_delta", rul_now - rul_then) or (rul_now - rul_then))

        def _build_plain_explanation(main_feature: str) -> str:
            if trend == "decreasing":
                return (
                    f"{machine_name} has about {rul_now:.0f} cycles left, down from {rul_then:.0f} in this selected period. "
                    f"The strongest warning signal is {main_feature}, which means wear is building faster than before. "
                    "If this continues, unplanned downtime risk increases, so plan intervention in the next maintenance window."
                )
            if trend == "increasing":
                return (
                    f"{machine_name} has about {rul_now:.0f} cycles left, up from {rul_then:.0f} in this selected period. "
                    f"The strongest current signal is {main_feature}, and overall condition is improving for now. "
                    "Continue routine checks so this recovery trend remains stable."
                )
            return (
                f"{machine_name} is steady at about {rul_now:.0f} cycles in this selected period. "
                f"The strongest current signal is {main_feature}, but no clear acceleration is detected right now. "
                "Continue scheduled checks and escalate if the remaining life starts dropping again."
            )

        def _normalize_output(payload: Dict[str, Any]) -> Dict[str, Any]:
            text = str(payload.get("explanation", "")).strip()
            contradiction = False
            if trend == "increasing" and re.search(r"decreas|drop|worse|issue", text, re.IGNORECASE):
                contradiction = True
            if trend == "decreasing" and re.search(r"going up|increas|improv", text, re.IGNORECASE):
                contradiction = True

            main_feature = top[0].get("feature", "equipment load") if top else "equipment load"
            if contradiction or not text:
                text = _build_plain_explanation(main_feature)
            elif len(text.split()) < 18:
                # If model output is too short to be useful, provide an operator-ready summary.
                text = _build_plain_explanation(main_feature)

            payload["explanation"] = text

            if not payload.get("actions"):
                if trend == "increasing":
                    payload["actions"] = [{
                        "inspect_first": top[0].get("feature", "sensor signal") if top else "sensor signal",
                        "check_now": "Confirm readings remain in normal range.",
                        "plan_by": "Routine check this week",
                        "escalate_if": "RUL starts dropping for two consecutive checks.",
                    }]
                elif trend == "decreasing":
                    payload["actions"] = [{
                        "inspect_first": top[0].get("feature", "sensor signal") if top else "sensor signal",
                        "check_now": "Verify abnormal readings against threshold.",
                        "plan_by": "Within 72 hours",
                        "escalate_if": "RUL keeps dropping quickly or status turns red.",
                    }]
                else:
                    payload["actions"] = [{
                        "inspect_first": "general condition",
                        "check_now": "Continue normal monitoring.",
                        "plan_by": "Next scheduled maintenance",
                        "escalate_if": "A clear downward RUL trend appears.",
                    }]

            return payload

        if not self.llm:
            top_text = ", ".join([str(item.get("feature", "unknown")) for item in top[:2]]) or "no dominant drivers"
            risk_band = structured_facts.get("risk_band", "Unknown")
            anomalies = structured_facts.get("anomalies", [])
            main_feature = top[0].get("feature", "equipment load") if top else "equipment load"
            actions = [
                {
                    "inspect_first": top[0].get("feature", "vibration") if top else "vibration",
                    "check_now": "Verify trend against threshold and current operating load.",
                    "plan_by": "Within 24 hours" if risk_band == "Red" else "Within 72 hours",
                    "escalate_if": "RUL slope worsens or risk band becomes Red.",
                }
            ]
            return {
                "explanation": (
                    f"{machine_name} has about {rul_now:.0f} cycles left ({delta:+.0f} change in this selected period). "
                    f"Main signal to watch is {main_feature}. Risk level is {risk_band}. "
                    "If the drop continues over the next checks, move from monitoring to scheduled intervention."
                ),
                "actions": actions,
                "confidence": structured_facts.get("confidence", "low"),
                "evidence_notes": anomalies,
                "source_mode": "fallback_structured_facts",
            }

        prompt = PromptTemplate.from_template(
            """
            You are a maintenance advisor for non-technical factory operators.

            RULES:
            - Use ONLY the structured facts provided below.
            - Do not invent sensor values, hidden causes, or external context.
                        - Keep explanation practical, plain language, and evidence-linked.
                        - Write for non-technical operators (no jargon such as SHAP, slope, trendline, regression, harmonics).
                        - Keep total explanation under 70 words.
                        - Use short sentences.
                                                - Structure explanation in this order:
                                                    1) what changed (RUL now vs before),
                                                    2) why it matters for operations,
                                                    3) what the team should do next.
                        - Be direction-consistent:
                            - If rul.trend_direction is "decreasing", explain RUL is going down.
                            - If rul.trend_direction is "increasing", explain RUL is going up.
                            - If rul.trend_direction is "stable", explain RUL is stable.
                        - Never say "decreasing" when rul_then < rul_now.
                        - Never say "increasing" when rul_then > rul_now.
                        - When trend_direction is "increasing", avoid alarmist words like "issue" or "worse".
            - Output valid JSON only.

            MACHINE: {machine_name}
            CURRENT_RUL: {rul}

            STRUCTURED_FACTS_JSON:
            {facts_json}

            Return exactly this JSON schema:
            {{
                            "explanation": "2-4 short plain-language sentences for operators.",
              "actions": [
                {{
                                    "inspect_first": "simple part name",
                                    "check_now": "one concrete check in plain language",
                                    "plan_by": "simple deadline like 'today' or 'within 72 hours'",
                                    "escalate_if": "simple trigger condition"
                }}
              ],
              "confidence": "low|medium|high",
              "evidence_notes": ["short evidence bullet", "short evidence bullet"],
              "source_mode": "structured_facts_only"
            }}
            """
        )

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "machine_name": machine_name,
                    "rul": rul,
                    "facts_json": facts_json,
                }
            )
            content = str(response.content).strip()
            json_match = re.search(r"(\{.*\})", content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            parsed_data = json.loads(content)
            parsed_data.setdefault("explanation", "No explanation generated.")
            parsed_data.setdefault("actions", [])
            parsed_data.setdefault("confidence", structured_facts.get("confidence", "low"))
            parsed_data.setdefault("evidence_notes", structured_facts.get("anomalies", []))
            parsed_data.setdefault("source_mode", "structured_facts_only")
            return _normalize_output(parsed_data)
        except Exception as e:
            return _normalize_output({
                "explanation": f"Structured-facts explanation failed: {str(e)}",
                "actions": [],
                "confidence": structured_facts.get("confidence", "low"),
                "evidence_notes": structured_facts.get("anomalies", []),
                "source_mode": "structured_facts_error",
            })

    async def generate_rul_maintenance_plans(
        self,
        machine_name: str,
        rul: int,
        structured_facts: Dict[str, Any],
        rag_context: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate three operator-focused maintenance strategies (Urgent, Balanced,
        Deferred) from replay structured facts, optionally enriched with RAG context.
        """
        rag_context = rag_context or []
        facts_json = json.dumps(structured_facts, ensure_ascii=True)
        context_text = "\n\n".join([str(c.get("content", "")) for c in rag_context if c.get("content")])
        rag_used = bool(context_text.strip())
        top_drivers = structured_facts.get("top_drivers", []) or []
        trend = str(structured_facts.get("rul", {}).get("trend_direction", "stable")).lower()
        risk_band = str(structured_facts.get("risk_band", "Unknown"))

        def _fallback_plan(plan_type: str, plan_by: str, risk_level: str, team: str, cost: str) -> Dict[str, Any]:
            first_driver = top_drivers[0].get("feature", "vibration") if top_drivers else "vibration"
            return {
                "plan_type": plan_type,
                "headline": f"{plan_type.capitalize()} maintenance plan for {machine_name}",
                "summary": f"Focus first on {first_driver} and confirm trend before next production window.",
                "inspect_first": first_driver,
                "check_now": "Compare latest readings against recent baseline and operating load.",
                "plan_by": plan_by,
                "escalate_if": "RUL drops for two consecutive checks or risk band turns Red.",
                "estimated_downtime": "2-6 hours",
                "estimated_cost": cost,
                "labor": team,
                "risk_level": risk_level,
                "confidence": structured_facts.get("confidence", "low"),
            }

        fallback_payload = {
            "strategies": [
                _fallback_plan("urgent", "Within 24 hours", "high", "2 technicians", "$4,500-$9,000"),
                _fallback_plan("balanced", "Within 72 hours", "medium", "1-2 technicians", "$2,500-$6,000"),
                _fallback_plan("deferred", "Next scheduled maintenance window", "medium-high", "1 technician", "$1,500-$4,000"),
            ],
            "source_mode": "fallback_structured_facts",
            "rag_used": rag_used,
            "notes": structured_facts.get("anomalies", []),
        }

        if not self.llm:
            return fallback_payload

        prompt = PromptTemplate.from_template(
            """
            You are a senior maintenance planner creating actionable strategy options.

            RULES:
            - Use ONLY the structured facts and optional context supplied below.
            - Return valid JSON only.
            - Provide exactly 3 strategies: urgent, balanced, deferred.
            - Keep each field concise and practical for operators.
            - If trend_direction is increasing, avoid alarmist tone.
            - Include realistic estimates for downtime, labor, and cost ranges.

            MACHINE: {machine_name}
            CURRENT_RUL: {rul}

            STRUCTURED_FACTS_JSON:
            {facts_json}

            OPTIONAL_RAG_CONTEXT:
            {context_text}

            Return exactly this JSON schema:
            {{
              "strategies": [
                {{
                  "plan_type": "urgent",
                  "headline": "string",
                  "summary": "string",
                  "inspect_first": "string",
                  "check_now": "string",
                  "plan_by": "string",
                  "escalate_if": "string",
                  "estimated_downtime": "string",
                  "estimated_cost": "string",
                  "labor": "string",
                  "risk_level": "high|medium|low",
                  "confidence": "low|medium|high"
                }},
                {{
                  "plan_type": "balanced",
                  "headline": "string",
                  "summary": "string",
                  "inspect_first": "string",
                  "check_now": "string",
                  "plan_by": "string",
                  "escalate_if": "string",
                  "estimated_downtime": "string",
                  "estimated_cost": "string",
                  "labor": "string",
                  "risk_level": "high|medium|low",
                  "confidence": "low|medium|high"
                }},
                {{
                  "plan_type": "deferred",
                  "headline": "string",
                  "summary": "string",
                  "inspect_first": "string",
                  "check_now": "string",
                  "plan_by": "string",
                  "escalate_if": "string",
                  "estimated_downtime": "string",
                  "estimated_cost": "string",
                  "labor": "string",
                  "risk_level": "high|medium|low",
                  "confidence": "low|medium|high"
                }}
              ]
            }}
            """
        )

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "machine_name": machine_name,
                    "rul": rul,
                    "facts_json": facts_json,
                    "context_text": context_text or "No external context available.",
                }
            )
            content = str(response.content).strip()
            json_match = re.search(r"(\{.*\})", content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            parsed_data = json.loads(content)
            strategies = parsed_data.get("strategies", [])
            if not isinstance(strategies, list) or len(strategies) != 3:
                return fallback_payload

            normalized: List[Dict[str, Any]] = []
            for idx, strategy in enumerate(strategies):
                if not isinstance(strategy, dict):
                    continue
                mode = str(strategy.get("plan_type", "")).strip().lower()
                if not mode:
                    mode = ["urgent", "balanced", "deferred"][min(idx, 2)]
                normalized.append(
                    {
                        "plan_type": mode,
                        "headline": str(strategy.get("headline", f"{mode.capitalize()} maintenance plan")).strip(),
                        "summary": str(strategy.get("summary", "No summary provided.")).strip(),
                        "inspect_first": str(strategy.get("inspect_first", top_drivers[0].get("feature", "vibration") if top_drivers else "vibration")).strip(),
                        "check_now": str(strategy.get("check_now", "Verify trend against latest telemetry.")).strip(),
                        "plan_by": str(strategy.get("plan_by", "Within 72 hours")).strip(),
                        "escalate_if": str(strategy.get("escalate_if", "RUL trend worsens on next checks.")).strip(),
                        "estimated_downtime": str(strategy.get("estimated_downtime", "TBD")).strip(),
                        "estimated_cost": str(strategy.get("estimated_cost", "TBD")).strip(),
                        "labor": str(strategy.get("labor", "TBD")).strip(),
                        "risk_level": str(strategy.get("risk_level", risk_band)).strip().lower(),
                        "confidence": str(strategy.get("confidence", structured_facts.get("confidence", "low"))).strip().lower(),
                    }
                )

            if len(normalized) != 3:
                return fallback_payload

            return {
                "strategies": normalized,
                "source_mode": "rag_enriched_structured_facts" if rag_used else "structured_facts_only",
                "rag_used": rag_used,
                "notes": structured_facts.get("anomalies", []),
            }
        except Exception:
            # Ensure endpoint remains stable even when LLM output is malformed.
            source_mode = "fallback_rag_error" if rag_used else "fallback_structured_facts_error"
            fallback_payload["source_mode"] = source_mode
            if trend == "increasing":
                fallback_payload["notes"] = list(fallback_payload.get("notes", [])) + ["Trend currently improving; keep normal monitoring cadence."]
            return fallback_payload


reasoning_service = ReasoningService()
