import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from services.database import supabase
from datetime import datetime
from pathlib import Path
from services import ml_predictor
from utils.logger import log_error

class MLService:
    """
    Service to handle ML model interactions for RUL prediction.
    This acts as an interface to the actual ML model developed by the teammate.
    """

    def __init__(self):
        self.models_dir = str((Path(__file__).resolve().parents[1] / "models" / "saved"))

    def _sensor_aliases(self) -> Dict[str, list]:
        return {
            "vibration": ["vibration", "vib", "sensor_vibration", "sensor1", "sensor_1", "s1"],
            "temperature": ["temperature", "temp", "sensor_temperature", "sensor2", "sensor_2", "s2"],
            "load": ["load", "pressure", "torque", "sensor_load", "sensor3", "sensor_3", "s3"],
        }

    def _build_sensor_frame(self, csv_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize arbitrary CSV columns into [vibration, temperature, load]."""
        if csv_data.empty:
            raise ValueError("Uploaded CSV is empty.")

        lower_to_original = {str(col).strip().lower(): col for col in csv_data.columns}
        aliases = self._sensor_aliases()

        # First, try explicit/alias-based matching.
        selected = {}
        for feature_name, alias_list in aliases.items():
            for alias in alias_list:
                if alias in lower_to_original:
                    selected[feature_name] = lower_to_original[alias]
                    break

        ignored_columns = {
            "unit_id",
            "unit",
            "id",
            "cycle",
            "time",
            "timestamp",
            "date",
            "recorded_at",
        }
        candidate_columns = [
            c
            for c in csv_data.columns
            if str(c).strip().lower() not in ignored_columns
        ]
        numeric_df = csv_data[candidate_columns].apply(pd.to_numeric, errors="coerce") if candidate_columns else pd.DataFrame()

        # Fallback: greedily choose informative numeric columns for missing features.
        if len(selected) < 3:
            available_cols = [
                c
                for c in numeric_df.columns
                if c not in selected.values() and numeric_df[c].notna().sum() > 0
            ]

            def _feature_score(series: pd.Series, feature_name: str) -> float:
                clean = series.dropna()
                if clean.empty:
                    return -1e9
                std = float(clean.std())
                median = float(clean.median())
                # Heuristic priors for expected scales.
                priors = {
                    "vibration": (0.0, 5.0),
                    "temperature": (20.0, 200.0),
                    "load": (20.0, 300.0),
                }
                lo, hi = priors[feature_name]
                in_range_bonus = 1.0 if lo <= median <= hi else 0.0
                return std + in_range_bonus

            for feature_name in ["vibration", "temperature", "load"]:
                if feature_name in selected:
                    continue
                if not available_cols:
                    break
                best_col = max(available_cols, key=lambda c: _feature_score(numeric_df[c], feature_name))
                selected[feature_name] = best_col
                available_cols.remove(best_col)

        missing = [f for f in ["vibration", "temperature", "load"] if f not in selected]
        if missing:
            raise ValueError(
                "Could not map required sensor features (vibration, temperature, load). "
                f"Missing: {', '.join(missing)}"
            )

        sensor_df = pd.DataFrame(
            {
                "vibration": pd.to_numeric(csv_data[selected["vibration"]], errors="coerce"),
                "temperature": pd.to_numeric(csv_data[selected["temperature"]], errors="coerce"),
                "load": pd.to_numeric(csv_data[selected["load"]], errors="coerce"),
            }
        )
        sensor_df = sensor_df.dropna(how="all")

        if sensor_df.empty:
            raise ValueError("Sensor CSV must contain numeric values.")

        sensor_df = sensor_df.ffill().bfill()
        if sensor_df.isna().any().any():
            raise ValueError("CSV has missing values that cannot be resolved.")

        return sensor_df

    def _prepare_sensor_matrix(self, csv_data: pd.DataFrame, dataset_id: str = "FD001") -> np.ndarray:
        """Build a numeric matrix for model inference and validate minimum shape."""
        numeric_df = self._build_sensor_frame(csv_data)
        numeric_df = self._normalize_sensor_frame(numeric_df, dataset_id=dataset_id)

        if len(numeric_df) < 3:
            raise ValueError("At least 3 sensor rows are required for stable prediction.")

        return numeric_df.values.astype(np.float32)

    def _scale_to_range(self, series: pd.Series, out_lo: float, out_hi: float) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        clean = values.dropna()
        if clean.empty:
            return values

        q_low = float(clean.quantile(0.05))
        q_high = float(clean.quantile(0.95))
        if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
            return pd.Series(np.full(len(values), (out_lo + out_hi) / 2.0), index=values.index)

        clipped = values.clip(lower=q_low, upper=q_high)
        scaled = (clipped - q_low) / (q_high - q_low)
        return out_lo + scaled * (out_hi - out_lo)

    def _normalize_sensor_frame(self, sensor_df: pd.DataFrame, dataset_id: str = "FD001") -> pd.DataFrame:
        """Map upload sensors into the same normalized ranges used by replay inference."""
        normalized = sensor_df.copy()
        targets = {
            "vibration": (0.2, 2.0),
            "temperature": (45.0, 95.0),
            "load": (70.0, 150.0),
        }

        reference_stats = None
        try:
            from services.replay_service import replay_service

            reference_stats = replay_service.get_reference_scaler_stats(dataset_id)
        except Exception:
            reference_stats = None

        for col, (lo, hi) in targets.items():
            series = pd.to_numeric(normalized[col], errors="coerce")
            if reference_stats and col in reference_stats:
                src_lo, src_hi = reference_stats[col]
                clipped = series.clip(lower=src_lo, upper=src_hi)
                scaled = (clipped - src_lo) / max(src_hi - src_lo, 1e-6)
                normalized[col] = lo + scaled * (hi - lo)
                continue

            in_range_ratio = float(series.between(lo, hi, inclusive="both").mean()) if len(series) else 0.0
            normalized[col] = series if in_range_ratio >= 0.85 else self._scale_to_range(series, lo, hi)

        return normalized

    def _status_from_health_state(self, health_state: str, rul: int) -> str:
        normalized = (health_state or "").strip().lower()
        if normalized == "critical":
            return "Red"
        if normalized == "warning":
            return "Yellow"
        if normalized == "healthy":
            return "Green"
        return self.determine_status(rul)

    def _persist_sensor_history(self, machine_id: int, csv_data: pd.DataFrame, source: str = "upload"):
        try:
            history_df = self._build_sensor_frame(csv_data)
        except Exception as exc:
            # Persistence should not block prediction flow.
            log_error("2", f"Sensor history persistence skipped for machine_id={machine_id}: {str(exc)}")
            return

        rows = []
        base_time = datetime.now()
        for idx, row in history_df.tail(60).iterrows():
            rows.append(
                {
                    "machine_id": machine_id,
                    "source": source,
                    "operating_mode": "normal",
                    "vibration": None if pd.isna(row["vibration"]) else float(row["vibration"]),
                    "temperature": None if pd.isna(row["temperature"]) else float(row["temperature"]),
                    "load": None if pd.isna(row["load"]) else float(row["load"]),
                    "anomaly_score": 0.0,
                    "recorded_at": base_time.isoformat(),
                }
            )

        if rows:
            supabase.table("sensor_readings").insert(rows).execute()

    def predict_rul(self, csv_data: pd.DataFrame, machine_id: int, dataset_id: str = "FD001") -> Dict[str, Any]:
        """
        Predict Remaining Useful Life (RUL) from sensor data using the exported pipeline.
        """
        matrix = self._prepare_sensor_matrix(csv_data, dataset_id=dataset_id)
        prediction = ml_predictor.run_prediction(
            unit_id=str(machine_id),
            dataset_id=dataset_id,
            readings=matrix.tolist(),
            models_dir=self.models_dir,
        )

        return prediction

    def determine_status(self, rul: int) -> str:
        """Determines the Red/Yellow/Green status based on RUL."""
        if rul < 50:
            return "Red"
        elif rul < 150:
            return "Yellow"
        else:
            return "Green"

    async def update_machine_from_data(self, machine_id: int, file_path: str, dataset_id: str = "FD001") -> Dict[str, Any]:
        """
        Reads sensor data, predicts RUL, and updates the database.
        """
        # 1. Load data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {str(e)}")

        # 2. Persist upload into sensor history for trend visualization.
        self._persist_sensor_history(machine_id, df, source="upload")

        # 3. Predict RUL
        prediction = self.predict_rul(df, machine_id=machine_id, dataset_id=dataset_id)
        predicted_rul = int(round(float(prediction["rul_prediction"])))
        status = self._status_from_health_state(prediction.get("health_state", ""), predicted_rul)

        # 4. Update machine snapshot.
        update_data = {
            "current_rul": predicted_rul,
            "status": status,
            "last_updated": datetime.now().isoformat()
        }
        
        response = supabase.table("machines").update(update_data).eq("id", machine_id).execute()
        
        if not response.data:
            raise ValueError(f"Machine with ID {machine_id} not found.")

        # 5. Persist prediction history.
        supabase.table("prediction_history").insert(
            {
                "machine_id": machine_id,
                "source": "upload",
                "dataset_id": prediction["dataset_id"],
                "rul_prediction": float(prediction["rul_prediction"]),
                "health_state": prediction["health_state"],
                "status": status,
                "change_point_detected": prediction["change_point_detected"],
                "change_point_step": prediction["change_point_step"],
                "explanation": prediction["explanation"],
                "predicted_at": update_data["last_updated"],
            }
        ).execute()

        try:
            from services.explainability_service import explainability_service

            explainability_service.compute_for_machine(
                machine_id=machine_id,
                dataset_id=prediction["dataset_id"],
                window_size=60,
                source="upload",
                force_recompute=True,
                models_dir=self.models_dir,
            )
        except Exception as exc:
            # Explanation persistence should not block the primary prediction flow.
            log_error("2", f"Explainability persistence failed for machine_id={machine_id}: {str(exc)}")
            
        return {
            "machine_id": machine_id,
            "predicted_rul": predicted_rul,
            "status": status,
            "updated_at": update_data["last_updated"],
            "dataset_id": prediction["dataset_id"],
            "health_state": prediction["health_state"],
            "change_point_detected": prediction["change_point_detected"],
            "change_point_step": prediction["change_point_step"],
            "explanation": prediction["explanation"],
        }

ml_service = MLService()
