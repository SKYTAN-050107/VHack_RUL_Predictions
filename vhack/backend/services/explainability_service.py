from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.database import supabase
from services import ml_predictor
from utils.logger import log_action, log_error

try:
    import shap  # type: ignore
except Exception:
    shap = None


class ExplainabilityService:
    def __init__(self):
        self._cache: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

    def _latest_prediction(self, machine_id: int) -> Optional[Dict[str, Any]]:
        response = (
            supabase.table("prediction_history")
            .select("id,machine_id,source,dataset_id,rul_prediction,predicted_at")
            .eq("machine_id", machine_id)
            .order("predicted_at", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]
        return None

    def _latest_window(self, machine_id: int, window_size: int = 60) -> pd.DataFrame:
        response = (
            supabase.table("sensor_readings")
            .select("vibration,temperature,load,recorded_at")
            .eq("machine_id", machine_id)
            .order("recorded_at", desc=True)
            .limit(window_size)
            .execute()
        )
        rows = response.data or []
        if len(rows) < 3:
            raise ValueError("Insufficient sensor history for explainability. Need at least 3 rows.")

        df = pd.DataFrame(rows)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df = df.sort_values("recorded_at", ascending=True)
        numeric_df = df[["vibration", "temperature", "load"]].apply(pd.to_numeric, errors="coerce").ffill().bfill()
        if numeric_df.isna().any().any():
            raise ValueError("Sensor data has unresolved missing values for SHAP.")
        return numeric_df

    def _resolve_estimator(self, model: Any) -> Optional[Any]:
        if hasattr(model, "predict") and hasattr(model, "feature_importances_"):
            return model

        # Common containers for sklearn-style estimators
        for attr in ["model", "regressor", "estimator", "pipeline", "rul_model"]:
            nested = getattr(model, attr, None)
            if nested is not None and hasattr(nested, "predict"):
                if hasattr(nested, "feature_importances_"):
                    return nested
                model = nested

        # Last step in pipeline style object
        try:
            if hasattr(model, "steps") and model.steps:
                last_step = model.steps[-1][1]
                if hasattr(last_step, "predict"):
                    return last_step
        except Exception:
            pass

        return model if hasattr(model, "predict") else None

    def _predict_scalar(self, model: Any, x_matrix: np.ndarray) -> float:
        result = model.predict(x_matrix)
        if isinstance(result, dict):
            return float(result.get("rul_prediction", 0.0))
        return float(np.ravel(result)[-1])

    def _heuristic_shap(self, x_window: np.ndarray, feature_names: List[str], model_output: float) -> Dict[str, Any]:
        latest = x_window[-1]
        baseline = np.mean(x_window, axis=0)
        diffs = latest - baseline
        norm = float(np.sum(np.abs(diffs))) or 1.0
        shap_values = (diffs / norm) * abs(model_output)

        ranked = sorted(
            [
                {
                    "feature": feature_names[i],
                    "shap_value": round(float(shap_values[i]), 4),
                    "direction": "increase_risk" if shap_values[i] > 0 else "decrease_risk",
                }
                for i in range(len(feature_names))
            ],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )

        for idx, item in enumerate(ranked, start=1):
            item["rank"] = idx

        return {
            "mode": "heuristic",
            "base_value": float(np.mean(baseline)),
            "model_output": float(model_output),
            "top_features": ranked[:5],
            "full_values": {feature_names[i]: float(shap_values[i]) for i in range(len(feature_names))},
        }

    def _compute_shap(self, model: Any, x_window: np.ndarray, feature_names: List[str], model_output: float) -> Dict[str, Any]:
        if shap is None:
            return self._heuristic_shap(x_window, feature_names, model_output)

        estimator = self._resolve_estimator(model)
        if estimator is None:
            return self._heuristic_shap(x_window, feature_names, model_output)

        x_latest = x_window[-1:].astype(np.float32)
        x_background = x_window[-min(30, len(x_window)):].astype(np.float32)

        try:
            if hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
                raw_values = explainer.shap_values(x_latest)
                base_value = explainer.expected_value
            else:
                return self._heuristic_shap(x_window, feature_names, model_output)

            values = np.array(raw_values)
            if values.ndim == 3:
                values = values[0]
            values = np.ravel(values)

            ranked = sorted(
                [
                    {
                        "feature": feature_names[i],
                        "shap_value": round(float(values[i]), 4),
                        "direction": "increase_risk" if values[i] > 0 else "decrease_risk",
                    }
                    for i in range(min(len(feature_names), len(values)))
                ],
                key=lambda x: abs(x["shap_value"]),
                reverse=True,
            )
            for idx, item in enumerate(ranked, start=1):
                item["rank"] = idx

            base_scalar = float(np.ravel(np.array(base_value))[0]) if base_value is not None else 0.0
            return {
                "mode": "shap_tree",
                "base_value": base_scalar,
                "model_output": float(model_output),
                "top_features": ranked[:5],
                "full_values": {
                    feature_names[i]: float(values[i])
                    for i in range(min(len(feature_names), len(values)))
                },
            }
        except Exception as exc:
            log_error("3", f"SHAP computation failed, using heuristic fallback: {str(exc)}")
            return self._heuristic_shap(x_window, feature_names, model_output)

    def _persist(self, machine_id: int, dataset_id: str, prediction_id: Optional[int], source: str, shap_payload: Dict[str, Any]):
        supabase.table("shap_explanations").insert(
            {
                "machine_id": machine_id,
                "prediction_id": prediction_id,
                "source": source,
                "model_type": "joblib",
                "dataset_id": dataset_id,
                "base_value": float(shap_payload.get("base_value", 0.0)),
                "model_output": float(shap_payload.get("model_output", 0.0)),
                "top_features": shap_payload.get("top_features", []),
                "full_values": shap_payload.get("full_values", {}),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()

    def _parse_iso(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            return None

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _rul_trend_metrics(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not points:
            return {
                "points": 0,
                "rul_now": 0.0,
                "rul_then": 0.0,
                "rul_delta": 0.0,
                "rul_delta_pct": 0.0,
                "hourly_slope": 0.0,
            }

        sorted_points = sorted(
            points,
            key=lambda p: self._parse_iso(str(p.get("predicted_at", ""))) or datetime.min.replace(tzinfo=timezone.utc),
        )
        first = sorted_points[0]
        last = sorted_points[-1]
        rul_then = self._safe_float(first.get("rul_prediction"))
        rul_now = self._safe_float(last.get("rul_prediction"))
        rul_delta = rul_now - rul_then
        rul_delta_pct = (rul_delta / rul_then * 100.0) if rul_then else 0.0

        first_time = self._parse_iso(str(first.get("predicted_at", "")))
        last_time = self._parse_iso(str(last.get("predicted_at", "")))
        if first_time and last_time and last_time > first_time:
            hours = max((last_time - first_time).total_seconds() / 3600.0, 1e-6)
            hourly_slope = rul_delta / hours
        else:
            hourly_slope = 0.0

        return {
            "points": len(sorted_points),
            "rul_now": round(rul_now, 2),
            "rul_then": round(rul_then, 2),
            "rul_delta": round(rul_delta, 2),
            "rul_delta_pct": round(rul_delta_pct, 2),
            "hourly_slope": round(hourly_slope, 4),
        }

    def _risk_band(self, rul_now: float, hourly_slope: float) -> str:
        if rul_now <= 20 or hourly_slope <= -0.5:
            return "Red"
        if rul_now <= 60 or hourly_slope <= -0.2:
            return "Yellow"
        return "Green"

    def compute_driver_trend(
        self,
        machine_id: int,
        hours_lookback: int = 24,
        top_n: int = 5,
        dataset_id: str = "FD001",
    ) -> Dict[str, Any]:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_lookback)
        dataset = dataset_id.upper()

        prediction_rows = (
            supabase.table("prediction_history")
            .select("rul_prediction,predicted_at,status,health_state")
            .eq("machine_id", machine_id)
            .eq("dataset_id", dataset)
            .gte("predicted_at", start_time.isoformat())
            .order("predicted_at", desc=False)
            .execute()
        ).data or []

        shap_rows = (
            supabase.table("shap_explanations")
            .select("top_features,created_at")
            .eq("machine_id", machine_id)
            .eq("dataset_id", dataset)
            .gte("created_at", start_time.isoformat())
            .order("created_at", desc=False)
            .execute()
        ).data or []

        feature_map: Dict[str, Dict[str, Any]] = {}
        for row in shap_rows:
            created_at = row.get("created_at")
            for item in row.get("top_features") or []:
                feature = str(item.get("feature", "unknown"))
                shap_value = self._safe_float(item.get("shap_value"))
                direction = str(item.get("direction", "mixed"))
                rank = self._safe_int(item.get("rank"), default=999)

                if feature not in feature_map:
                    feature_map[feature] = {
                        "feature": feature,
                        "count": 0,
                        "sum_value": 0.0,
                        "max_abs": 0.0,
                        "directions": [],
                        "timeline": [],
                    }

                bucket = feature_map[feature]
                bucket["count"] += 1
                bucket["sum_value"] += shap_value
                bucket["max_abs"] = max(bucket["max_abs"], abs(shap_value))
                bucket["directions"].append(direction)
                bucket["timeline"].append(
                    {
                        "timestamp": created_at,
                        "shap_value": round(shap_value, 4),
                        "rank": rank,
                    }
                )

        ranked_drivers = sorted(
            feature_map.values(),
            key=lambda item: (item["count"], abs(item["sum_value"]) / max(item["count"], 1)),
            reverse=True,
        )

        top_drivers: List[Dict[str, Any]] = []
        for idx, item in enumerate(ranked_drivers[:top_n], start=1):
            avg_value = item["sum_value"] / max(item["count"], 1)
            direction_votes = Counter(item["directions"])
            consensus = direction_votes.most_common(1)[0][0] if direction_votes else "mixed"
            top_drivers.append(
                {
                    "feature": item["feature"],
                    "rank": idx,
                    "occurrence_count": item["count"],
                    "occurrence_pct": round((item["count"] / max(len(shap_rows), 1)) * 100.0, 2),
                    "avg_shap_value": round(avg_value, 4),
                    "max_abs_shap_value": round(item["max_abs"], 4),
                    "direction": consensus,
                    "timeline": item["timeline"],
                }
            )

        rul_metrics = self._rul_trend_metrics(prediction_rows)
        risk_band = self._risk_band(rul_metrics["rul_now"], rul_metrics["hourly_slope"])

        anomalies: List[str] = []
        if rul_metrics["rul_delta"] < -10:
            anomalies.append("RUL dropped by more than 10 units in lookback window")
        if rul_metrics["hourly_slope"] <= -0.5:
            anomalies.append("RUL decline slope indicates rapid degradation")
        if top_drivers and top_drivers[0]["occurrence_pct"] >= 60:
            anomalies.append("Single driver dominates degradation pattern")

        structured_facts = {
            "machine_id": machine_id,
            "dataset_id": dataset,
            "period": {
                "hours_lookback": hours_lookback,
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "rul": rul_metrics,
            "risk_band": risk_band,
            "confidence": "low" if len(shap_rows) < 3 else "medium" if len(shap_rows) < 8 else "high",
            "top_drivers": [
                {
                    "feature": d["feature"],
                    "rank": d["rank"],
                    "occurrence_pct": d["occurrence_pct"],
                    "avg_shap_value": d["avg_shap_value"],
                    "direction": d["direction"],
                }
                for d in top_drivers
            ],
            "anomalies": anomalies,
        }

        return {
            "status": "ok",
            "machine_id": machine_id,
            "dataset_id": dataset,
            "period_hours": hours_lookback,
            "predictions_analyzed": len(prediction_rows),
            "shap_records_analyzed": len(shap_rows),
            "top_drivers": top_drivers,
            "structured_facts": structured_facts,
        }

    def facts_summary_for_prompt(self, structured_facts: Dict[str, Any]) -> str:
        rul = structured_facts.get("rul", {})
        top = structured_facts.get("top_drivers", [])[:3]
        anomalies = structured_facts.get("anomalies", [])
        lines = [
            f"- Risk band: {structured_facts.get('risk_band', 'Unknown')}",
            f"- RUL now: {rul.get('rul_now', 0)}",
            f"- RUL change: {rul.get('rul_delta', 0)} ({rul.get('rul_delta_pct', 0)}%)",
            f"- Hourly slope: {rul.get('hourly_slope', 0)}",
        ]
        for item in top:
            lines.append(
                "- Driver {rank}: {feature}, occurrence={occurrence_pct}%, avg_shap={avg_shap_value}, direction={direction}".format(
                    rank=item.get("rank", 0),
                    feature=item.get("feature", "unknown"),
                    occurrence_pct=item.get("occurrence_pct", 0),
                    avg_shap_value=item.get("avg_shap_value", 0),
                    direction=item.get("direction", "mixed"),
                )
            )
        if anomalies:
            lines.append("- Anomalies: " + "; ".join(anomalies))
        else:
            lines.append("- Anomalies: none detected")
        return "\n".join(lines)

    def shap_summary_for_prompt(self, shap_payload: Dict[str, Any]) -> str:
        top = shap_payload.get("top_features", [])[:3]
        if not top:
            return "SHAP evidence unavailable."

        lines = []
        for item in top:
            feat = item.get("feature", "unknown")
            val = item.get("shap_value", 0)
            direction = item.get("direction", "neutral")
            lines.append(f"- {feat}: {val} ({direction})")
        return "\n".join(lines)

    def compute_for_machine(
        self,
        machine_id: int,
        dataset_id: str = "FD001",
        window_size: int = 60,
        source: str = "analysis",
        force_recompute: bool = False,
        models_dir: str = "models/saved",
    ) -> Dict[str, Any]:
        prediction = self._latest_prediction(machine_id)
        cache_key = (
            machine_id,
            (prediction or {}).get("predicted_at", "none"),
            dataset_id.upper(),
        )

        if not force_recompute and cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cache_hit"] = True
            return cached

        df = self._latest_window(machine_id, window_size=window_size)
        x_window = df.values.astype(np.float32)
        model = ml_predictor.load_pipeline(dataset_id, models_dir=models_dir)
        model_output = self._predict_scalar(model, x_window)

        feature_names = list(df.columns)
        shap_payload = self._compute_shap(model, x_window, feature_names, model_output)

        self._persist(
            machine_id=machine_id,
            dataset_id=dataset_id.upper(),
            prediction_id=(prediction or {}).get("id"),
            source=source,
            shap_payload=shap_payload,
        )

        response = {
            "status": "ok",
            "cache_hit": False,
            "mode": shap_payload.get("mode", "heuristic"),
            "base_value": shap_payload.get("base_value", 0.0),
            "model_output": shap_payload.get("model_output", model_output),
            "top_features": shap_payload.get("top_features", []),
            "dataset_id": dataset_id.upper(),
        }
        self._cache[cache_key] = response
        log_action("3", "SHAP explanation generated", f"Machine {machine_id}, mode={response['mode']}")
        return response


explainability_service = ExplainabilityService()
