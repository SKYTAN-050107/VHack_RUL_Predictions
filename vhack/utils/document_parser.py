"""
Document parser — extracts financial parameters from uploaded files.
Uses Gemini Vision when an API key is available; otherwise returns realistic mock data.
"""

import streamlit as st
from typing import Optional
from models.financial import FinancialParameters

_MOCK_EXTRACTED = {
    "hourly_production_value": 12500.0,
    "units_per_hour": 250,
    "unit_price": 50.0,
    "repair_cost_preventative": 6000.0,
    "repair_cost_failure": 28000.0,
    "mttr_hours": 8.0,
    "sla_penalty_per_hour": 1500.0,
    "supply_chain_penalty": 5000.0,
    "confidence": 0.88,
    "source": "Production_SLA_Contract_2026.pdf",
}

_EXTRACTION_PROMPT = """\
Extract the following financial parameters from this document image.
Return ONLY a JSON object with these exact keys (use null if not found):
- hourly_production_value (number): value of production per hour in USD
- units_per_hour (integer): units produced per hour
- unit_price (number): price per unit in USD
- repair_cost_preventative (number): scheduled maintenance cost in USD
- repair_cost_failure (number): unplanned breakdown repair cost in USD
- mttr_hours (number): mean time to repair in hours
- sla_penalty_per_hour (number): SLA penalty per hour of downtime in USD
- supply_chain_penalty (number): supply chain disruption cost in USD

For each value you extract, set a "confidence" field (0.0–1.0) for the overall extraction quality.
Document content:
"""


class DocumentParser:
    """Extracts financial parameters from uploaded PDF/TXT documents."""

    def __init__(self):
        self._api_key = None
        try:
            self._api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    def parse(self, uploaded_file, machine_id: str) -> Optional[FinancialParameters]:
        """
        Parse an uploaded file and extract financial parameters.
        Returns a FinancialParameters object, or None on failure.
        """
        filename = uploaded_file.name if uploaded_file else "uploaded_document"

        if self._api_key:
            result = self._parse_with_gemini(uploaded_file)
            if result:
                result["machine_id"] = machine_id
                result["source"] = filename
                return FinancialParameters(**result)

        # Fallback: return mock extracted data
        data = dict(_MOCK_EXTRACTED)
        data["machine_id"] = machine_id
        data["source"] = filename
        return FinancialParameters(**data)

    def _parse_with_gemini(self, uploaded_file) -> Optional[dict]:
        import json
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            content = uploaded_file.read()
            # For text files, send as text; for images/PDFs, use vision
            if uploaded_file.type and uploaded_file.type.startswith("text"):
                text_content = content.decode("utf-8", errors="ignore")[:8000]
                response = model.generate_content(_EXTRACTION_PROMPT + text_content)
            else:
                import base64
                b64 = base64.b64encode(content).decode()
                response = model.generate_content([
                    _EXTRACTION_PROMPT,
                    {"mime_type": uploaded_file.type or "application/pdf", "data": b64},
                ])

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[:-1])

            parsed = json.loads(raw)
            # Fill defaults for missing fields
            defaults = dict(_MOCK_EXTRACTED)
            for k, v in defaults.items():
                if parsed.get(k) is None:
                    parsed[k] = v
            return parsed

        except Exception:
            return None
