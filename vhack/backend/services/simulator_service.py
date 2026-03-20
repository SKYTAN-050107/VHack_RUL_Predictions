import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services.database import get_supabase
from services.ml_service import ml_service
from utils.logger import log_action, log_error


class SimulatorService:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._interval_seconds = 4
        self._tick = 0
        self._seed = 42
        self._last_tick_at: Optional[str] = None
        self._sim_db = None
        self._profile_cache: Dict[int, Dict[str, Any]] = {}
        self._window_cache: Dict[int, List[Dict[str, Any]]] = {}

    def start(self, interval_seconds: int = 4, seed: int = 42) -> Dict[str, Any]:
        with self._lock:
            if self._running:
                return self.status()

            self._interval_seconds = max(1, int(interval_seconds))
            self._seed = int(seed)
            self._stop_event.clear()
            self._running = True
            self._sim_db = get_supabase()

            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

        log_action("8", "Simulator started", f"interval={self._interval_seconds}s seed={self._seed}")
        return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._running:
                return self.status()

            self._stop_event.set()
            self._running = False

        log_action("8", "Simulator stop requested")
        return self.status()

    def step_once(self) -> Dict[str, Any]:
        updated = self._tick_once()
        return {
            "message": "Simulator step executed",
            "updated_machines": updated,
            **self.status(),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "interval_seconds": self._interval_seconds,
            "tick": self._tick,
            "seed": self._seed,
            "last_tick_at": self._last_tick_at,
        }

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception as exc:
                log_error("8", f"Simulator tick failed: {str(exc)}")

            self._stop_event.wait(self._interval_seconds)

    def _db(self):
        if self._sim_db is None:
            self._sim_db = get_supabase()
        return self._sim_db

    def _with_retry(self, func, retries: int = 3, delay_seconds: float = 0.12):
        last_exc = None
        for attempt in range(retries):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(delay_seconds * (attempt + 1))
        raise last_exc

    def _fetch_machines(self) -> List[Dict[str, Any]]:
        response = self._with_retry(
            lambda: self._db().table("machines").select("id,name,type,current_rul,status").execute()
        )
        return response.data or []

    def _get_or_create_profile(self, machine: Dict[str, Any]) -> Dict[str, Any]:
        machine_id = machine["id"]
        if machine_id in self._profile_cache:
            return self._profile_cache[machine_id]

        profile_resp = self._with_retry(
            lambda: self._db()
            .table("simulation_profiles")
            .select("*")
            .eq("machine_id", machine_id)
            .limit(1)
            .execute()
        )
        if profile_resp.data:
            self._profile_cache[machine_id] = profile_resp.data[0]
            return profile_resp.data[0]

        # Default profile for machine type.
        machine_type = str(machine.get("type", "Generic")).lower()
        base_map = {
            "pump": (0.7, 58.0, 95.0),
            "conveyor": (0.5, 52.0, 80.0),
            "press": (0.9, 65.0, 120.0),
        }
        base_vibration, base_temperature, base_load = base_map.get(machine_type, (0.6, 55.0, 90.0))

        payload = {
            "machine_id": machine_id,
            "base_vibration": base_vibration,
            "base_temperature": base_temperature,
            "base_load": base_load,
            "wear_rate": 0.004,
            "anomaly_probability": 0.04,
            "dataset_id": "FD001",
        }
        created = self._with_retry(lambda: self._db().table("simulation_profiles").insert(payload).execute())
        self._profile_cache[machine_id] = created.data[0]
        return created.data[0]

    def _resolve_mode(self, status: str, rng: np.random.Generator) -> str:
        status_norm = (status or "").lower()
        if status_norm == "red":
            return "stressed"

        p = rng.random()
        if p < 0.1:
            return "idle"
        if p < 0.25:
            return "warmup"
        if p < 0.8:
            return "normal"
        return "stressed"

    def _simulate_reading(self, machine: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        machine_id = machine["id"]
        rng = np.random.default_rng(self._seed + machine_id * 100_000 + self._tick)

        current_rul = int(machine.get("current_rul") or 400)
        health_ratio = max(0.0, min(1.0, current_rul / 500.0))
        degradation = 1.0 - health_ratio

        mode = self._resolve_mode(machine.get("status", "Green"), rng)
        mode_scale = {
            "idle": (0.75, 0.95, 0.65),
            "warmup": (1.05, 1.10, 0.90),
            "normal": (1.0, 1.0, 1.0),
            "stressed": (1.25, 1.18, 1.25),
            "maintenance": (0.70, 0.85, 0.50),
        }[mode]

        wear_rate = float(profile.get("wear_rate") or 0.004)
        anomaly_probability = float(profile.get("anomaly_probability") or 0.04)

        vibration = float(profile["base_vibration"]) * mode_scale[0] + degradation * (0.4 + wear_rate * self._tick)
        temperature = float(profile["base_temperature"]) * mode_scale[1] + degradation * (9.0 + wear_rate * 100 * self._tick)
        load = float(profile["base_load"]) * mode_scale[2] + degradation * (12.0 + wear_rate * 80 * self._tick)

        vibration += rng.normal(0, 0.03)
        temperature += rng.normal(0, 0.9)
        load += rng.normal(0, 1.8)

        anomaly_score = 0.0
        if rng.random() < (anomaly_probability + degradation * 0.08):
            vibration += rng.uniform(0.18, 0.35)
            temperature += rng.uniform(1.5, 4.2)
            load += rng.uniform(3.0, 8.0)
            anomaly_score = float(rng.uniform(0.65, 0.98))

        return {
            "machine_id": machine_id,
            "operating_mode": mode,
            "vibration": round(max(0.01, vibration), 4),
            "temperature": round(max(1.0, temperature), 3),
            "load": round(max(1.0, load), 3),
            "anomaly_score": round(anomaly_score, 4),
            "source": "simulator",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": str(profile.get("dataset_id") or "FD001").upper(),
        }

    def _persist_sensor_row(self, row: Dict[str, Any]):
        self._with_retry(lambda: self._db().table("sensor_readings").insert(
            {
                "machine_id": row["machine_id"],
                "source": row["source"],
                "operating_mode": row["operating_mode"],
                "vibration": row["vibration"],
                "temperature": row["temperature"],
                "load": row["load"],
                "anomaly_score": row["anomaly_score"],
                "recorded_at": row["recorded_at"],
            }
        ).execute())

    def _append_window(self, machine_id: int, row: Dict[str, Any], window_size: int = 60):
        bucket = self._window_cache.setdefault(machine_id, [])
        bucket.append(
            {
                "vibration": row["vibration"],
                "temperature": row["temperature"],
                "load": row["load"],
                "recorded_at": row["recorded_at"],
            }
        )
        if len(bucket) > window_size:
            del bucket[:-window_size]

    def _predict_from_recent_history(self, machine_id: int, dataset_id: str) -> Optional[Dict[str, Any]]:
        rows = self._window_cache.get(machine_id, [])
        if not rows:
            history = self._with_retry(
                lambda: self._db()
                .table("sensor_readings")
                .select("vibration,temperature,load,recorded_at")
                .eq("machine_id", machine_id)
                .order("recorded_at", desc=True)
                .limit(60)
                .execute()
            )
            rows = list(reversed(history.data or []))
            self._window_cache[machine_id] = rows

        if len(rows) < 3:
            # Warm-up phase: keep collecting telemetry until minimum window is available.
            return None

        df = pd.DataFrame(rows)
        if "recorded_at" in df.columns:
            df = df.sort_values("recorded_at", ascending=True)
        df = df[["vibration", "temperature", "load"]]

        prediction = ml_service.predict_rul(df, machine_id=machine_id, dataset_id=dataset_id)
        predicted_rul = int(round(float(prediction["rul_prediction"])))
        status = ml_service._status_from_health_state(prediction.get("health_state", ""), predicted_rul)

        now_iso = datetime.now(timezone.utc).isoformat()
        self._with_retry(lambda: self._db().table("machines").update(
            {
                "current_rul": predicted_rul,
                "status": status,
                "last_updated": now_iso,
            }
        ).eq("id", machine_id).execute())

        self._with_retry(lambda: self._db().table("prediction_history").insert(
            {
                "machine_id": machine_id,
                "source": "simulator",
                "dataset_id": prediction["dataset_id"],
                "rul_prediction": float(prediction["rul_prediction"]),
                "health_state": prediction["health_state"],
                "status": status,
                "change_point_detected": prediction["change_point_detected"],
                "change_point_step": prediction["change_point_step"],
                "explanation": prediction["explanation"],
                "predicted_at": now_iso,
            }
        ).execute())

        return {
            "machine_id": machine_id,
            "predicted_rul": predicted_rul,
            "status": status,
            "health_state": prediction["health_state"],
            "dataset_id": prediction["dataset_id"],
        }

    def _tick_once(self) -> int:
        machines = self._fetch_machines()
        if not machines:
            return 0

        updated = 0
        for machine in machines:
            try:
                profile = self._get_or_create_profile(machine)
                reading = self._simulate_reading(machine, profile)
                self._persist_sensor_row(reading)
                self._append_window(machine["id"], reading)
                prediction = self._predict_from_recent_history(machine["id"], reading["dataset_id"])
                if prediction is None and self._tick < 3:
                    log_action("8", "Simulator warm-up", f"Machine {machine['id']} awaiting minimum history")
                updated += 1
            except Exception as exc:
                log_error("8", f"Machine simulation failed (id={machine.get('id')}): {str(exc)}")

        self._tick += 1
        self._last_tick_at = datetime.now(timezone.utc).isoformat()
        if updated:
            log_action("8", "Simulator tick completed", f"tick={self._tick} updated={updated}")
        return updated


simulator_service = SimulatorService()