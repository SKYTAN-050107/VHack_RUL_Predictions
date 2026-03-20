"""Lightweight backend API client for root Streamlit pages."""

import json
from typing import Any, Dict
from urllib import error, parse, request


class BackendAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _get_json(self, path: str, query: Dict[str, Any]) -> Dict[str, Any]:
        query_string = parse.urlencode(query)
        url = f"{self.base_url}{path}?{query_string}" if query_string else f"{self.base_url}{path}"
        req = request.Request(url=url, method="GET")
        with request.urlopen(req, timeout=12) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}

    def fetch_driver_trend(
        self,
        machine_id: int,
        hours_lookback: int = 24,
        top_n: int = 5,
        dataset_id: str = "FD001",
    ) -> Dict[str, Any]:
        try:
            return self._get_json(
                path=f"/api/machines/{machine_id}/driver-trend",
                query={
                    "hours_lookback": hours_lookback,
                    "top_n": top_n,
                    "dataset_id": dataset_id,
                },
            )
        except error.HTTPError as exc:
            return {
                "status": "error",
                "detail": f"HTTP {exc.code}: failed to fetch driver trend",
            }
        except Exception as exc:
            return {
                "status": "error",
                "detail": str(exc),
            }
