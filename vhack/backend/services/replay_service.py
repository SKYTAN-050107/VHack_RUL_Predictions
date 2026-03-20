from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd

from services import ml_predictor


@dataclass
class ReplayPoint:
    cycle: int
    vibration: float
    temperature: float
    load: float
    predicted_rul: float
    actual_rul: float
    abs_error: float


class ReplayService:
    """Serve C-MAPSS replay slices and aligned prediction/ground-truth traces."""

    def __init__(self) -> None:
        backend_root = Path(__file__).resolve().parents[1]
        self.data_dir = backend_root / "data" / "cmapss"
        self.models_dir = str(backend_root / "models" / "saved")
        self.min_window = 3
        self._shap_cache: Dict[Tuple[str, str, int, str, int, int], Dict[str, object]] = {}
        self._replay_points_cache: Dict[Tuple[str, str, int, str, int, int], List[Dict[str, object]]] = {}
        self._reference_scaler_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def _canonical_dataset_id(self, dataset_id: str) -> str:
        key = (dataset_id or "FD001").upper().strip()
        if key not in {"FD001", "FD002", "FD003", "FD004"}:
            raise ValueError("dataset_id must be one of FD001, FD002, FD003, FD004")
        return key

    def _dataset_file_candidates(self, dataset_id: str, split: str) -> List[Path]:
        split_key = split.lower().strip()
        if split_key not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        did = self._canonical_dataset_id(dataset_id)
        upper = did.upper()
        lower = did.lower()
        if split_key == "train":
            return [
                self.data_dir / f"train_{upper}.txt",
                self.data_dir / f"train_{lower}.txt",
                self.data_dir / f"Train_{upper}.txt",
                self.data_dir / f"train_{upper}.csv",
            ]
        return [
            self.data_dir / f"test_{upper}.txt",
            self.data_dir / f"test_{lower}.txt",
            self.data_dir / f"Test_{upper}.txt",
            self.data_dir / f"test_{upper}.csv",
        ]

    def _rul_file_candidates(self, dataset_id: str) -> List[Path]:
        did = self._canonical_dataset_id(dataset_id)
        upper = did.upper()
        lower = did.lower()
        return [
            self.data_dir / f"RUL_{upper}.txt",
            self.data_dir / f"rul_{upper}.txt",
            self.data_dir / f"RUL_{lower}.txt",
        ]

    def _resolve_sensor_mapping(self, columns: List[str]) -> Dict[str, str]:
        feature_candidates = [c for c in columns if str(c).startswith("s")]
        if len(feature_candidates) < 3:
            raise ValueError("C-MAPSS split has fewer than 3 sensor channels")

        return {
            "vibration": "s2" if "s2" in columns else feature_candidates[0],
            "temperature": "s7" if "s7" in columns else feature_candidates[1],
            "load": "s12" if "s12" in columns else feature_candidates[2],
        }

    def _get_reference_scaler_stats(self, dataset_id: str) -> Dict[str, Tuple[float, float]]:
        did = self._canonical_dataset_id(dataset_id)
        if did in self._reference_scaler_stats:
            return self._reference_scaler_stats[did]

        train_file = next((p for p in self._dataset_file_candidates(did, "train") if p.exists()), None)
        if train_file is None:
            raise FileNotFoundError(
                f"Train split not found for {did}. Expected under {self.data_dir}."
            )

        train_raw = pd.read_csv(train_file, sep=r"\s+", header=None, engine="python")
        if train_raw.shape[1] < 6:
            raise ValueError(f"Unexpected C-MAPSS shape in {train_file.name}: {train_raw.shape}")

        column_names = ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
            f"s{i}" for i in range(1, train_raw.shape[1] - 4)
        ]
        train_raw.columns = column_names

        for col in train_raw.columns:
            train_raw[col] = pd.to_numeric(train_raw[col], errors="coerce")

        mapping = self._resolve_sensor_mapping(list(train_raw.columns))
        stats: Dict[str, Tuple[float, float]] = {}
        for feature, source_col in mapping.items():
            clean = pd.to_numeric(train_raw[source_col], errors="coerce").dropna()
            if clean.empty:
                stats[feature] = (0.0, 1.0)
            else:
                q_low = float(clean.quantile(0.05))
                q_high = float(clean.quantile(0.95))
                if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
                    median = float(clean.median())
                    stats[feature] = (median - 0.5, median + 0.5)
                else:
                    stats[feature] = (q_low, q_high)

        self._reference_scaler_stats[did] = stats
        return stats

    def get_reference_scaler_stats(self, dataset_id: str) -> Dict[str, Tuple[float, float]]:
        """Public accessor for deterministic, train-derived normalization bounds."""
        return self._get_reference_scaler_stats(dataset_id)

    @lru_cache(maxsize=16)
    def _load_split(self, dataset_id: str, split: str) -> pd.DataFrame:
        source_file = next((p for p in self._dataset_file_candidates(dataset_id, split) if p.exists()), None)
        if source_file is None:
            raise FileNotFoundError(
                f"C-MAPSS file not found for {dataset_id} {split}. "
                f"Expected under {self.data_dir}."
            )

        # C-MAPSS uses whitespace-delimited values with occasional trailing spaces.
        raw = pd.read_csv(source_file, sep=r"\s+", header=None, engine="python")
        if raw.shape[1] < 6:
            raise ValueError(f"Unexpected C-MAPSS shape in {source_file.name}: {raw.shape}")

        column_names = ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
            f"s{i}" for i in range(1, raw.shape[1] - 4)
        ]
        raw.columns = column_names

        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        raw = raw.dropna(subset=["unit_id", "cycle"]).copy()
        raw["unit_id"] = raw["unit_id"].astype(int)
        raw["cycle"] = raw["cycle"].astype(int)

        if split.lower().strip() == "train":
            max_cycle = raw.groupby("unit_id")["cycle"].transform("max")
            raw["actual_rul"] = (max_cycle - raw["cycle"]).astype(float)
        else:
            rul_file = next((p for p in self._rul_file_candidates(dataset_id) if p.exists()), None)
            if rul_file is None:
                raise FileNotFoundError(
                    f"RUL label file not found for {dataset_id} test split. "
                    f"Expected RUL_{dataset_id}.txt under {self.data_dir}."
                )

            rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, engine="python")
            rul_df = rul_df.dropna(how="all")
            rul_df = rul_df.rename(columns={0: "final_rul"})
            rul_df["unit_id"] = range(1, len(rul_df) + 1)
            rul_df["final_rul"] = pd.to_numeric(rul_df["final_rul"], errors="coerce")

            max_cycle_df = raw.groupby("unit_id", as_index=False)["cycle"].max().rename(columns={"cycle": "max_cycle"})
            raw = raw.merge(max_cycle_df, on="unit_id", how="left")
            raw = raw.merge(rul_df[["unit_id", "final_rul"]], on="unit_id", how="left")

            if raw["final_rul"].isna().any():
                raise ValueError("Missing final_rul labels for one or more units in test split")

            # Off-by-one-safe reconstruction: cycle axis is taken directly from file values.
            raw["actual_rul"] = (raw["max_cycle"] - raw["cycle"] + raw["final_rul"]).astype(float)
            raw = raw.drop(columns=["max_cycle", "final_rul"])

        # Stable feature mapping to project sensor schema.
        mapping = self._resolve_sensor_mapping(list(raw.columns))
        reference_stats = self._get_reference_scaler_stats(dataset_id)

        raw["vibration_raw"] = pd.to_numeric(raw[mapping["vibration"]], errors="coerce")
        raw["temperature_raw"] = pd.to_numeric(raw[mapping["temperature"]], errors="coerce")
        raw["load_raw"] = pd.to_numeric(raw[mapping["load"]], errors="coerce")

        # Normalize to the profile expected by existing predictor exports:
        # vibration ~ [0.2, 2.0], temperature ~ [45, 95], load ~ [70, 150].
        # Use train-derived reference quantiles for deterministic, leakage-safe replay scaling.
        vib_lo, vib_hi = reference_stats["vibration"]
        tmp_lo, tmp_hi = reference_stats["temperature"]
        load_lo, load_hi = reference_stats["load"]
        raw["vibration"] = self._scale_to_range(raw["vibration_raw"], 0.2, 2.0, q_low=vib_lo, q_high=vib_hi)
        raw["temperature"] = self._scale_to_range(raw["temperature_raw"], 45.0, 95.0, q_low=tmp_lo, q_high=tmp_hi)
        raw["load"] = self._scale_to_range(raw["load_raw"], 70.0, 150.0, q_low=load_lo, q_high=load_hi)

        raw = raw.dropna(subset=["vibration", "temperature", "load", "actual_rul"]).copy()
        raw["dataset_id"] = self._canonical_dataset_id(dataset_id)
        raw["split"] = split.lower().strip()
        raw = raw.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        return raw

    def _scale_to_range(
        self,
        series: pd.Series,
        out_lo: float,
        out_hi: float,
        q_low: float | None = None,
        q_high: float | None = None,
    ) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        clean = values.dropna()
        if clean.empty:
            return values

        lo = float(clean.quantile(0.05)) if q_low is None else float(q_low)
        hi = float(clean.quantile(0.95)) if q_high is None else float(q_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # Keep a sensible midpoint if the channel is near-constant.
            return pd.Series(np.full(len(values), (out_lo + out_hi) / 2.0), index=values.index)

        clipped = values.clip(lower=lo, upper=hi)
        scaled = (clipped - lo) / (hi - lo)
        return out_lo + scaled * (out_hi - out_lo)

    def list_units(self, dataset_id: str, split: str) -> List[int]:
        frame = self._load_split(dataset_id, split)
        return sorted(frame["unit_id"].astype(int).unique().tolist())

    def get_range_info(self, dataset_id: str, split: str, unit_id: int) -> Dict[str, object]:
        frame = self._load_split(dataset_id, split)
        unit_df = frame[frame["unit_id"] == int(unit_id)].copy()
        if unit_df.empty:
            raise ValueError(f"No rows for unit_id={unit_id} in {dataset_id} {split}")

        return {
            "dataset_id": self._canonical_dataset_id(dataset_id),
            "split": split.lower().strip(),
            "unit_id": int(unit_id),
            "range_start": int(unit_df["cycle"].min()),
            "range_end": int(unit_df["cycle"].max()),
            "points_in_range": int(len(unit_df)),
            "available_units": self.list_units(dataset_id, split),
            "model_modes": ml_predictor.get_model_mode_status(models_dir=self.models_dir),
        }

    def _slice_unit(self, dataset_id: str, split: str, unit_id: int, start_cycle: int, end_cycle: int) -> pd.DataFrame:
        frame = self._load_split(dataset_id, split)
        unit_df = frame[frame["unit_id"] == int(unit_id)].copy()
        if unit_df.empty:
            raise ValueError(f"No rows for unit_id={unit_id} in {dataset_id} {split}")

        min_cycle = int(unit_df["cycle"].min())
        max_cycle = int(unit_df["cycle"].max())
        lo = max(min_cycle, int(start_cycle))
        hi = min(max_cycle, int(end_cycle))
        if lo > hi:
            return unit_df.iloc[0:0].copy()

        return unit_df[(unit_df["cycle"] >= lo) & (unit_df["cycle"] <= hi)].copy().sort_values("cycle")

    def build_replay_payload(
        self,
        dataset_id: str,
        split: str,
        unit_id: int,
        model_mode: str,
        start_cycle: int,
        end_cycle: int,
        telemetry_limit: int = 60,
        shap_enabled: bool = False,
        shap_sample_interval: int = 10,
    ) -> Dict[str, object]:
        unit_slice = self._slice_unit(dataset_id, split, unit_id, start_cycle, end_cycle)
        shap_interval = max(1, int(shap_sample_interval))
        shap_timeline: Dict[str, Dict[str, object]] = {}
        shap_hits = 0
        shap_misses = 0
        shap_start = time.perf_counter()
        shap_note = "disabled"
        if unit_slice.empty:
            return {
                "metadata": {
                    "dataset_id": self._canonical_dataset_id(dataset_id),
                    "split": split.lower().strip(),
                    "unit_id": int(unit_id),
                    "model_mode": model_mode.lower().strip(),
                    "range_start": int(start_cycle),
                    "range_end": int(end_cycle),
                    "points_in_range": 0,
                },
                "series": [],
                "kpi": None,
                "shap_timeline": {},
                "shap_meta": {
                    "enabled": bool(shap_enabled),
                    "sample_interval": shap_interval,
                    "computed_cycles": [],
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "compute_ms": 0,
                    "note": "No rows in selected replay range",
                },
            }

        mode = model_mode.lower().strip()
        sensor_frame = unit_slice[["cycle", "vibration", "temperature", "load", "actual_rul"]].reset_index(drop=True)

        cache_enabled = not bool(shap_enabled)
        cache_key = (
            self._canonical_dataset_id(dataset_id),
            split.lower().strip(),
            int(unit_id),
            mode,
            int(telemetry_limit),
            int(start_cycle),
        )

        points: List[Dict[str, object]] = []
        start_idx = 0
        if cache_enabled:
            cached = self._replay_points_cache.get(cache_key, [])
            if cached:
                # Reuse computed prefix for progressive slider expansion.
                capped = cached[: len(sensor_frame)]
                points = [dict(item) for item in capped]
                start_idx = len(points)

        for idx in range(start_idx, len(sensor_frame)):
            begin = max(0, idx - telemetry_limit + 1)
            window = sensor_frame.iloc[begin : idx + 1]
            if len(window) < self.min_window:
                # For the first cycles, pad with the earliest row so playback starts
                # from cycle 1 without returning an empty frame.
                first_row = window.iloc[[0]][["vibration", "temperature", "load"]]
                pad_count = self.min_window - len(window)
                padded = pd.concat([first_row] * pad_count + [window[["vibration", "temperature", "load"]]], ignore_index=True)
                readings = padded.values.tolist()
            else:
                readings = window[["vibration", "temperature", "load"]].values.tolist()

            pred = ml_predictor.run_prediction_with_mode(
                unit_id=str(unit_id),
                dataset_id=dataset_id,
                readings=readings,
                model_mode=mode,
                models_dir=self.models_dir,
            )

            cycle = int(window.iloc[-1]["cycle"])
            actual_rul = float(window.iloc[-1]["actual_rul"])
            predicted_rul = float(pred["rul_prediction"])
            abs_error = float(abs(predicted_rul - actual_rul))
            points.append(
                {
                    "cycle": cycle,
                    "vibration": float(window.iloc[-1]["vibration"]),
                    "temperature": float(window.iloc[-1]["temperature"]),
                    "load": float(window.iloc[-1]["load"]),
                    "predicted_rul": round(predicted_rul, 2),
                    "actual_rul": round(actual_rul, 2),
                    "abs_error": round(abs_error, 2),
                    "health_state": pred.get("health_state", "Unknown"),
                    "dataset_id": pred.get("dataset_id", self._canonical_dataset_id(dataset_id)),
                    "model_mode": mode,
                    "inference_mode": pred.get("inference_mode", "model"),
                    "fallback_used": bool(pred.get("fallback_used", False)),
                    "fallback_reason": pred.get("fallback_reason"),
                    "is_true_adaptation": bool(pred.get("is_true_adaptation", False)),
                }
            )

            should_compute_shap = bool(shap_enabled) and (cycle % shap_interval == 0 or idx == len(sensor_frame) - 1)
            if should_compute_shap:
                shap_key = (
                    self._canonical_dataset_id(dataset_id),
                    split.lower().strip(),
                    int(unit_id),
                    mode,
                    int(cycle),
                    int(telemetry_limit),
                )
                cached = self._shap_cache.get(shap_key)
                if cached is not None:
                    shap_timeline[str(cycle)] = cached
                    shap_hits += 1
                else:
                    shap_payload = self._compute_shap_payload(
                        readings=np.array(readings, dtype=np.float32),
                        model_output=float(predicted_rul),
                        mode=mode,
                    )
                    self._shap_cache[shap_key] = shap_payload
                    shap_timeline[str(cycle)] = shap_payload
                    shap_misses += 1

        if cache_enabled:
            self._replay_points_cache[cache_key] = [dict(item) for item in points]
            # Simple bounded cache to avoid unbounded memory growth.
            while len(self._replay_points_cache) > 64:
                oldest_key = next(iter(self._replay_points_cache))
                del self._replay_points_cache[oldest_key]

        kpi = None
        if points:
            latest = points[-1]
            prev = points[-2] if len(points) > 1 else None
            kpi = {
                "predicted_rul": latest["predicted_rul"],
                "actual_rul": latest["actual_rul"],
                "abs_error": latest["abs_error"],
                "predicted_delta": None if prev is None else round(float(latest["predicted_rul"] - prev["predicted_rul"]), 2),
                "actual_delta": None if prev is None else round(float(latest["actual_rul"] - prev["actual_rul"]), 2),
            }

        if shap_enabled:
            if mode == "adapted":
                shap_note = "SHAP explains base signal attribution; adapted-mode output includes post-hoc adjustment."
            else:
                shap_note = "SHAP sampled along replay timeline in normalized feature space."
        else:
            shap_note = "SHAP disabled"

        shap_ms = int((time.perf_counter() - shap_start) * 1000)

        return {
            "metadata": {
                "dataset_id": self._canonical_dataset_id(dataset_id),
                "split": split.lower().strip(),
                "unit_id": int(unit_id),
                "model_mode": mode,
                "range_start": int(sensor_frame["cycle"].min()),
                "range_end": int(sensor_frame["cycle"].max()),
                "points_in_range": int(len(points)),
            },
            "series": points,
            "kpi": kpi,
            "shap_timeline": shap_timeline,
            "shap_meta": {
                "enabled": bool(shap_enabled),
                "sample_interval": shap_interval,
                "computed_cycles": [int(k) for k in shap_timeline.keys()],
                "cache_hits": shap_hits,
                "cache_misses": shap_misses,
                "compute_ms": shap_ms,
                "note": shap_note,
            },
        }

    def _compute_shap_payload(self, readings: np.ndarray, model_output: float, mode: str) -> Dict[str, object]:
        feature_names = ["vibration", "temperature", "load"]
        if readings.ndim != 2 or readings.shape[0] == 0:
            return {
                "mode": "heuristic",
                "base_value": 0.0,
                "model_output": float(model_output),
                "top_features": [],
            }

        baseline = np.mean(readings, axis=0)
        latest = readings[-1]
        diffs = latest - baseline
        norm = float(np.sum(np.abs(diffs))) or 1.0
        shap_values = (diffs / norm) * abs(float(model_output))

        ranked = sorted(
            [
                {
                    "feature": feature_names[i],
                    "shap_value": round(float(shap_values[i]), 4),
                    "direction": "increase_risk" if shap_values[i] > 0 else "decrease_risk",
                }
                for i in range(min(len(feature_names), len(shap_values)))
            ],
            key=lambda x: abs(float(x["shap_value"])),
            reverse=True,
        )
        for idx, item in enumerate(ranked, start=1):
            item["rank"] = idx

        return {
            "mode": "heuristic",
            "base_value": round(float(np.mean(baseline)), 4),
            "model_output": round(float(model_output), 4),
            "top_features": ranked[:5],
            "note": "adapted-base-attribution" if mode == "adapted" else "base-attribution",
        }


replay_service = ReplayService()
