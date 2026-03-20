from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Dict, Any
import os
import asyncio
from services.database import supabase
from services.rag_service import rag_service
from services.reasoning_service import reasoning_service
from services.ml_service import ml_service
from services.ml_predictor import list_available_models, get_model_mode_status
from services.simulator_service import simulator_service
from services.explainability_service import explainability_service
from services.replay_service import replay_service
from models.database_models import Machine, MachineUpdate
from datetime import datetime
from utils.logger import log_action, log_error

router = APIRouter()


def _build_replay_structured_facts(replay_payload: Dict[str, Any]) -> Dict[str, Any]:
    series = replay_payload.get("series", []) or []
    shap_timeline = replay_payload.get("shap_timeline", {}) or {}

    if not series:
        return {
            "period": {"points": 0},
            "rul": {"rul_now": 0.0, "rul_then": 0.0, "rul_delta": 0.0, "rul_delta_pct": 0.0, "hourly_slope": 0.0},
            "risk_band": "Unknown",
            "confidence": "low",
            "top_drivers": [],
            "anomalies": ["No replay points available for this engine unit range."],
        }

    first = series[0]
    last = series[-1]
    rul_then = float(first.get("predicted_rul", 0.0))
    rul_now = float(last.get("predicted_rul", 0.0))
    rul_delta = rul_now - rul_then
    rul_delta_pct = (rul_delta / rul_then * 100.0) if rul_then else 0.0
    hourly_slope = rul_delta / max(len(series), 1)
    trend_direction = "decreasing" if rul_delta < 0 else "increasing" if rul_delta > 0 else "stable"

    feature_map: Dict[str, Dict[str, Any]] = {}
    for cycle_shap in shap_timeline.values():
        for item in (cycle_shap.get("top_features") or [])[:5]:
            feature = str(item.get("feature", "unknown"))
            value = float(item.get("shap_value", 0.0))
            direction = str(item.get("direction", "mixed"))
            if feature not in feature_map:
                feature_map[feature] = {
                    "count": 0,
                    "sum": 0.0,
                    "direction_votes": [],
                }
            feature_map[feature]["count"] += 1
            feature_map[feature]["sum"] += value
            feature_map[feature]["direction_votes"].append(direction)

    ranked = sorted(
        feature_map.items(),
        key=lambda kv: (kv[1]["count"], abs(kv[1]["sum"]) / max(kv[1]["count"], 1)),
        reverse=True,
    )

    top_drivers = []
    for idx, (feature, stats) in enumerate(ranked[:5], start=1):
        votes = stats["direction_votes"]
        consensus = max(set(votes), key=votes.count) if votes else "mixed"
        avg_shap = round(stats["sum"] / max(stats["count"], 1), 4)
        effect_on_rul = "raises_rul" if avg_shap > 0 else "lowers_rul" if avg_shap < 0 else "neutral"
        top_drivers.append(
            {
                "feature": feature,
                "rank": idx,
                "occurrence_count": int(stats["count"]),
                "occurrence_pct": round((stats["count"] / max(len(shap_timeline), 1)) * 100.0, 2),
                "avg_shap_value": avg_shap,
                "direction": consensus,
                "effect_on_rul": effect_on_rul,
            }
        )

    risk_band = "Green"
    if rul_now <= 20 or hourly_slope <= -0.5:
        risk_band = "Red"
    elif rul_now <= 60 or hourly_slope <= -0.2:
        risk_band = "Yellow"

    anomalies = []
    if rul_delta < -10:
        anomalies.append("RUL dropped sharply across selected replay window")
    if hourly_slope <= -0.5:
        anomalies.append("Rapid downward trend in predicted RUL")
    if top_drivers and top_drivers[0]["occurrence_pct"] >= 60:
        anomalies.append("One driver repeatedly dominates degradation signal")

    confidence = "low" if len(shap_timeline) < 3 else "medium" if len(shap_timeline) < 8 else "high"

    return {
        "period": {
            "points": len(series),
            "start_cycle": int(first.get("cycle", 0)),
            "end_cycle": int(last.get("cycle", 0)),
        },
        "rul": {
            "rul_now": round(rul_now, 2),
            "rul_then": round(rul_then, 2),
            "rul_delta": round(rul_delta, 2),
            "rul_delta_abs": round(abs(rul_delta), 2),
            "rul_delta_pct": round(rul_delta_pct, 2),
            "rul_delta_pct_abs": round(abs(rul_delta_pct), 2),
            "hourly_slope": round(hourly_slope, 4),
            "trend_direction": trend_direction,
        },
        "risk_band": risk_band,
        "confidence": confidence,
        "top_drivers": top_drivers,
        "anomalies": anomalies,
    }


@router.post("/simulator/start")
async def start_simulator(interval_seconds: int = Query(default=4, ge=1, le=60), seed: int = Query(default=42)):
    """Start background synthetic telemetry simulator."""
    try:
        return simulator_service.start(interval_seconds=interval_seconds, seed=seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulator/stop")
async def stop_simulator():
    """Stop background synthetic telemetry simulator."""
    try:
        return simulator_service.stop()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulator/step")
async def step_simulator_once():
    """Generate one simulation tick and update RUL once."""
    try:
        return simulator_service.step_once()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulator/status")
async def get_simulator_status():
    """Return current simulator status."""
    return simulator_service.status()


@router.get("/{machine_id}/telemetry")
async def get_machine_telemetry(machine_id: int, limit: int = Query(default=60, ge=5, le=500)):
    """Return latest telemetry points for a machine."""
    try:
        response = (
            supabase.table("sensor_readings")
            .select("machine_id,source,operating_mode,vibration,temperature,load,anomaly_score,recorded_at")
            .eq("machine_id", machine_id)
            .order("recorded_at", desc=True)
            .limit(limit)
            .execute()
        )
        data = list(reversed(response.data or []))
        return {"machine_id": machine_id, "points": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{machine_id}/rul-trend")
async def get_machine_rul_trend(machine_id: int, limit: int = Query(default=40, ge=5, le=500)):
    """Return latest RUL predictions for a machine."""
    try:
        response = (
            supabase.table("prediction_history")
            .select("machine_id,source,dataset_id,rul_prediction,health_state,status,change_point_detected,change_point_step,predicted_at")
            .eq("machine_id", machine_id)
            .order("predicted_at", desc=True)
            .limit(limit)
            .execute()
        )
        data = list(reversed(response.data or []))
        return {"machine_id": machine_id, "points": data}
    except Exception as e:
        log_error("8", f"Failed to fetch RUL trend for machine {machine_id}: {str(e)}")
        return {"machine_id": machine_id, "points": []}


@router.get("/dashboard/live")
async def get_live_dashboard(
    telemetry_limit: int = Query(default=30, ge=5, le=120),
    trend_limit: int = Query(default=20, ge=5, le=120),
):
    """Return machines + recent telemetry + RUL trends in one payload for low-latency dashboards."""
    try:
        machines_resp = supabase.table("machines").select("*").execute()
        machines = machines_resp.data or []
        machine_ids = [m["id"] for m in machines]

        telemetry_map = {mid: [] for mid in machine_ids}
        trend_map = {mid: [] for mid in machine_ids}

        for machine_id in machine_ids:
            telemetry_resp = (
                supabase.table("sensor_readings")
                .select("machine_id,source,operating_mode,vibration,temperature,load,anomaly_score,recorded_at")
                .eq("machine_id", machine_id)
                .order("recorded_at", desc=True)
                .limit(telemetry_limit)
                .execute()
            )
            telemetry_map[machine_id] = list(reversed(telemetry_resp.data or []))

            trend_resp = (
                supabase.table("prediction_history")
                .select("machine_id,source,dataset_id,rul_prediction,health_state,status,change_point_detected,change_point_step,predicted_at")
                .eq("machine_id", machine_id)
                .order("predicted_at", desc=True)
                .limit(trend_limit)
                .execute()
            )
            trend_map[machine_id] = list(reversed(trend_resp.data or []))

        return {
            "simulator": simulator_service.status(),
            "machines": machines,
            "telemetry": telemetry_map,
            "rul_trends": trend_map,
        }
    except Exception as e:
        log_error("8", f"Live dashboard fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/replay/range-info")
async def get_replay_range_info(
    dataset_id: str = Query(default="FD001", description="Dataset key (FD001-FD004)"),
    split: str = Query(default="test", description="Data split: train or test"),
    unit_id: int = Query(default=1, ge=1),
):
    """Return replay bounds and available units for one dataset/split."""
    try:
        return replay_service.get_range_info(dataset_id=dataset_id, split=split, unit_id=unit_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log_error("8", f"Replay range-info failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/dashboard/replay")
async def get_replay_dashboard(
    dataset_id: str = Query(default="FD001", description="Dataset key (FD001-FD004)"),
    split: str = Query(default="test", description="Data split: train or test"),
    unit_id: int = Query(default=1, ge=1),
    model_mode: str = Query(default="base", description="Model mode: base|adapted"),
    start_cycle: int = Query(default=1, ge=0),
    end_cycle: int = Query(default=99999, ge=0),
    telemetry_limit: int = Query(default=60, ge=5, le=200),
    shap_enabled: bool = Query(default=False, description="Enable SHAP timeline sampling"),
    shap_sample_interval: int = Query(default=10, ge=1, le=100, description="Compute SHAP every N cycles"),
):
    """Return cycle-aligned telemetry + predicted/actual RUL for replay dashboards."""
    try:
        return replay_service.build_replay_payload(
            dataset_id=dataset_id,
            split=split,
            unit_id=unit_id,
            model_mode=model_mode,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            telemetry_limit=telemetry_limit,
            shap_enabled=shap_enabled,
            shap_sample_interval=shap_sample_interval,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ADAPTED_MODE_UNAVAILABLE",
                "message": str(exc),
                "details": "Switch to base mode or install adapted-model runtime/artifacts.",
            },
        )
    except Exception as exc:
        log_error("8", f"Replay dashboard fetch failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/replay/driver-explanation")
async def get_replay_driver_explanation(
    dataset_id: str = Query(default="FD001", description="Dataset key (FD001-FD004)"),
    split: str = Query(default="test", description="Data split: train or test"),
    unit_id: int = Query(default=1, ge=1),
    model_mode: str = Query(default="base", description="Model mode: base|adapted"),
    start_cycle: int = Query(default=1, ge=0),
    end_cycle: int = Query(default=99999, ge=0),
    telemetry_limit: int = Query(default=60, ge=5, le=200),
    shap_sample_interval: int = Query(default=10, ge=1, le=100),
    include_maintenance: bool = Query(default=True, description="Include maintenance strategy recommendations"),
):
    """Generate plain-language RUL drop explanation for a replay engine unit from structured facts."""
    try:
        # Keep endpoint responsive: attempt SHAP facts first, then degrade gracefully.
        replay_payload = None
        try:
            replay_payload = await asyncio.wait_for(
                asyncio.to_thread(
                    replay_service.build_replay_payload,
                    dataset_id=dataset_id,
                    split=split,
                    unit_id=unit_id,
                    model_mode=model_mode,
                    start_cycle=start_cycle,
                    end_cycle=end_cycle,
                    telemetry_limit=telemetry_limit,
                    shap_enabled=True,
                    shap_sample_interval=shap_sample_interval,
                ),
                timeout=6.0,
            )
        except Exception as replay_exc:
            log_action("3", "Replay explanation SHAP path unavailable", str(replay_exc))
            try:
                replay_payload = await asyncio.wait_for(
                    asyncio.to_thread(
                        replay_service.build_replay_payload,
                        dataset_id=dataset_id,
                        split=split,
                        unit_id=unit_id,
                        model_mode=model_mode,
                        start_cycle=start_cycle,
                        end_cycle=end_cycle,
                        telemetry_limit=telemetry_limit,
                        shap_enabled=False,
                        shap_sample_interval=shap_sample_interval,
                    ),
                    timeout=10.0,
                )
            except Exception as fallback_exc:
                log_action("3", "Replay explanation non-SHAP fallback unavailable", str(fallback_exc))
                replay_payload = {
                    "series": [],
                    "shap_timeline": {},
                }

        structured_facts = _build_replay_structured_facts(replay_payload)
        machine_name = f"Engine Unit {unit_id} ({dataset_id.upper()} {split.lower()})"
        rul_now = int(structured_facts.get("rul", {}).get("rul_now", 0) or 0)

        async def _safe_driver_explanation() -> Dict[str, Any]:
            try:
                return await asyncio.wait_for(
                    reasoning_service.generate_rul_driver_explanation(
                        machine_name=machine_name,
                        rul=rul_now,
                        structured_facts=structured_facts,
                    ),
                    timeout=4.0,
                )
            except Exception:
                top = structured_facts.get("top_drivers", []) or []
                top_name = top[0].get("feature", "sensor signal") if top else "sensor signal"
                trend = structured_facts.get("rul", {}).get("trend_direction", "stable")
                rul_current = float(structured_facts.get("rul", {}).get("rul_now", rul_now) or rul_now)
                rul_previous = float(structured_facts.get("rul", {}).get("rul_then", rul_current) or rul_current)
                return {
                    "explanation": (
                        f"{machine_name} has about {rul_current:.0f} cycles left, compared with {rul_previous:.0f} earlier in this view. "
                        f"The main signal to inspect is {top_name}. "
                        "If this drop continues in the next checks, schedule maintenance to reduce downtime risk."
                    ),
                    "actions": [
                        {
                            "inspect_first": top_name,
                            "check_now": "Confirm latest reading stays within normal operating range.",
                            "plan_by": "Within 72 hours",
                            "escalate_if": "RUL drops again in the next two checks.",
                        }
                    ],
                    "confidence": structured_facts.get("confidence", "low"),
                    "evidence_notes": structured_facts.get("anomalies", []),
                    "source_mode": "router_timeout_fallback",
                }

        explanation = await _safe_driver_explanation()

        maintenance_plans = {
            "strategies": [],
            "source_mode": "disabled",
            "rag_used": False,
            "notes": [],
        }
        if include_maintenance:
            rag_context = []
            rag_source_mode = "structured_facts_only"
            try:
                top_features = [str(x.get("feature", "")).strip() for x in (structured_facts.get("top_drivers") or [])[:3]]
                top_features = [name for name in top_features if name]
                query = (
                    f"Maintenance actions for {dataset_id.upper()} engine unit {unit_id} in {split.lower()} split. "
                    f"Trend: {structured_facts.get('rul', {}).get('trend_direction', 'stable')}. "
                    f"Risk band: {structured_facts.get('risk_band', 'Unknown')}. "
                    f"Top drivers: {', '.join(top_features) if top_features else 'none'}."
                )
                rag_context = await asyncio.wait_for(rag_service.query_relevant_context(query, limit=4), timeout=1.5)
                rag_source_mode = "rag_enriched"
            except Exception as rag_exc:
                log_action("3", "RAG context unavailable for replay explanation", str(rag_exc))

            try:
                maintenance_plans = await asyncio.wait_for(
                    reasoning_service.generate_rul_maintenance_plans(
                        machine_name=machine_name,
                        rul=rul_now,
                        structured_facts=structured_facts,
                        rag_context=rag_context,
                    ),
                    timeout=4.5,
                )
            except Exception:
                maintenance_plans = {
                    "strategies": [],
                    "source_mode": "router_timeout_fallback",
                    "rag_used": False,
                    "notes": ["Maintenance plan generation timed out; retry for full strategies."],
                }

            if rag_source_mode == "structured_facts_only" and not maintenance_plans.get("rag_used", False):
                if str(maintenance_plans.get("source_mode", "")).startswith("fallback"):
                    maintenance_plans["source_mode"] = "fallback_structured_facts"
                elif maintenance_plans.get("source_mode") in {"structured_facts_only", "rag_enriched_structured_facts"}:
                    maintenance_plans["source_mode"] = "structured_facts_only"

        return {
            "status": "ok",
            "dataset_id": dataset_id.upper(),
            "split": split.lower(),
            "unit_id": int(unit_id),
            "model_mode": model_mode.lower(),
            "structured_facts": structured_facts,
            "llm_explanation": explanation,
            "maintenance_plans": maintenance_plans,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log_error("3", f"Replay driver explanation failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/")
async def get_machines():
    """Fetch all machines from Supabase or return mock data."""
    try:
        # Check if supabase client exists and is working
        if supabase:
            response = supabase.table("machines").select("*").execute()
            if response and hasattr(response, 'data'):
                return response.data
    except Exception as e:
        print(f"Supabase Access Error: {e}")
    
    # Return mock data for testing
    return [
        {"id": 1, "name": "Conveyor Belt A", "type": "Conveyor", "current_rul": 450, "status": "Green", "last_updated": datetime.now().isoformat()},
        {"id": 2, "name": "Motor Pump 03", "type": "Pump", "current_rul": 120, "status": "Yellow", "last_updated": datetime.now().isoformat()},
        {"id": 3, "name": "Hydraulic Press", "type": "Press", "current_rul": 15, "status": "Red", "last_updated": datetime.now().isoformat()}
    ]

@router.get("/{machine_id}")
async def get_machine(machine_id: int):
    """Fetch a single machine by ID."""
    response = supabase.table("machines").select("*").eq("id", machine_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Machine not found")
    return response.data

@router.get("/{machine_id}/analysis")
async def get_machine_analysis(
    machine_id: int,
    window_size: int = Query(default=60, ge=10, le=200),
    force_recompute: bool = Query(default=False),
):
    """
    Triggers Step 3, 4, and 5: LLM Root Cause analysis using RAG and Gemini.
    """
    # Step 3: Root Cause Reasoning
    log_action("3", "Starting Root Cause Reasoning", f"Machine ID: {machine_id}")
    
    try:
        # 1. Fetch machine data
        if supabase:
            machine_response = supabase.table("machines").select("*").eq("id", machine_id).single().execute()
            if machine_response and hasattr(machine_response, 'data'):
                machine = machine_response.data
            else:
                machine = {"id": machine_id, "name": f"Machine {machine_id}", "type": "Generic", "current_rul": 100}
        else:
            machine = {"id": machine_id, "name": f"Machine {machine_id}", "type": "Generic", "current_rul": 100}
    except Exception as e:
        log_action("5", "Error fetching machine for analysis", str(e))
        machine = {"id": machine_id, "name": f"Machine {machine_id}", "type": "Generic", "current_rul": 100}
    
    # Step 3: Map signals to possible component issues (RAG Retrieval)
    log_action("3", "Retrieving Technical Context", f"Machine Name: {machine.get('name')}")
    query = f"Failure modes and repair steps for {machine.get('name', 'Machine')} {machine.get('type', 'Generic')} with low RUL"
    context = await rag_service.query_relevant_context(query)
    
    # Fallback/Injection: Add mock files content if they exist to ground the LLM
    try:
        mock_files = ["backend/financial_data.txt", "backend/technical_specs.txt"]
        for mf in mock_files:
            if os.path.exists(mf):
                with open(mf, "r", encoding="utf-8") as f:
                    context.append({"content": f.read(), "metadata": {"source": mf}})
    except Exception as e:
        print(f"Error reading mock files: {e}")
    
    shap_payload = {
        "status": "unavailable",
        "cache_hit": False,
        "mode": "none",
        "base_value": 0.0,
        "model_output": 0.0,
        "dataset_id": "FD001",
        "top_features": [],
    }
    try:
        latest_prediction = (
            supabase.table("prediction_history")
            .select("dataset_id")
            .eq("machine_id", machine_id)
            .order("predicted_at", desc=True)
            .limit(1)
            .execute()
        )
        dataset_id = (latest_prediction.data or [{}])[0].get("dataset_id", "FD001")
        shap_payload = explainability_service.compute_for_machine(
            machine_id=machine_id,
            dataset_id=dataset_id,
            window_size=window_size,
            source="analysis",
            force_recompute=force_recompute,
            models_dir=ml_service.models_dir,
        )
    except Exception as e:
        log_action("3", "SHAP generation failed (continuing)", str(e))

    shap_summary = explainability_service.shap_summary_for_prompt(shap_payload)

    driver_trend_payload = {
        "status": "unavailable",
        "top_drivers": [],
        "structured_facts": {
            "machine_id": machine_id,
            "top_drivers": [],
            "anomalies": [],
            "confidence": "low",
            "risk_band": "Unknown",
            "rul": {
                "rul_now": machine.get("current_rul", 0),
                "rul_delta": 0,
                "rul_delta_pct": 0,
                "hourly_slope": 0,
            },
        },
    }
    driver_explanation = {
        "explanation": "Driver trend explanation unavailable.",
        "actions": [],
        "confidence": "low",
        "evidence_notes": [],
        "source_mode": "unavailable",
    }
    try:
        driver_trend_payload = explainability_service.compute_driver_trend(
            machine_id=machine_id,
            hours_lookback=24,
            top_n=5,
            dataset_id=shap_payload.get("dataset_id", "FD001"),
        )
        driver_explanation = await reasoning_service.generate_rul_driver_explanation(
            machine_name=machine.get("name", f"Machine {machine_id}"),
            rul=int(machine.get("current_rul", 0) or 0),
            structured_facts=driver_trend_payload.get("structured_facts", {}),
        )
    except Exception as e:
        log_action("3", "Driver trend explanation failed (continuing)", str(e))

    # Step 3 & 4: Root Cause Reasoning & Financial Risk analysis via Gemini
    log_action("4", "Calculating Financial & Operational Risk", "Analyzing downtime cost from context")
    analysis_data = await reasoning_service.generate_root_cause_analysis(
        machine.get("name", "Machine"),
        machine.get("current_rul", 100),
        context,
        shap_summary=shap_summary,
    )
    
    # Step 5: Management Report (Composite Response)
    log_action("5", "Generating Management Report", f"Analysis complete for {machine.get('name')}")
    
    # 4. Return the analysis and calculated financial data
    return {
        "machine_id": machine_id,
        "machine_name": machine.get("name", f"Machine {machine_id}"),
        "root_cause_analysis": analysis_data.get("analysis", "No analysis available."),
        "downtime_cost": analysis_data.get("downtime_risk_cost", 5000.00),
        "criticality_level": analysis_data.get("criticality_level", "Unknown"),
        "downtime_cost_breakdown": analysis_data.get("downtime_cost_breakdown", "Calculation details unavailable."),
        "discovery_path": analysis_data.get("discovery_path", "No discovery elaboration available."),
        "shap_explanation": shap_payload,
        "driver_trend": driver_trend_payload,
        "driver_trend_explanation": driver_explanation,
        "recommendations": analysis_data.get("recommendations", [
            {"label": "Replace Bearing", "priority": "Time"},
            {"label": "Calibrate Load Sensor", "priority": "Cost"},
            {"label": "Scheduled Maintenance", "priority": "Labor"}
        ])
    }


@router.get("/{machine_id}/driver-trend")
async def get_machine_driver_trend(
    machine_id: int,
    hours_lookback: int = Query(default=24, ge=1, le=168),
    top_n: int = Query(default=5, ge=1, le=10),
    dataset_id: str = Query(default="FD001"),
):
    """Return SHAP driver trend and an LLM explanation generated from structured facts only."""
    try:
        payload = explainability_service.compute_driver_trend(
            machine_id=machine_id,
            hours_lookback=hours_lookback,
            top_n=top_n,
            dataset_id=dataset_id,
        )

        machine_name = f"Machine {machine_id}"
        current_rul = int(payload.get("structured_facts", {}).get("rul", {}).get("rul_now", 0) or 0)
        try:
            machine_response = supabase.table("machines").select("name,current_rul").eq("id", machine_id).single().execute()
            if machine_response and getattr(machine_response, "data", None):
                machine_name = machine_response.data.get("name", machine_name)
                current_rul = int(machine_response.data.get("current_rul", current_rul) or current_rul)
        except Exception:
            pass

        explanation = await reasoning_service.generate_rul_driver_explanation(
            machine_name=machine_name,
            rul=current_rul,
            structured_facts=payload.get("structured_facts", {}),
        )
        return {
            **payload,
            "llm_explanation": explanation,
        }
    except Exception as e:
        log_error("3", f"Driver trend computation failed for machine {machine_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{machine_id}/upload-sensor-data")
async def upload_sensor_data(
    machine_id: int,
    file: UploadFile = File(...),
    dataset_id: str = Query(default="FD001", description="Dataset/model key (FD001-FD004)"),
):
    """
    Step 1 & 2: Upload sensor CSV, predict RUL, and convert to failure window.
    """
    log_action("1", "RUL Prediction triggered", f"Filename: {file.filename}")
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for sensor data.")
    
    # 1. Save file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        import shutil
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Step 1: ML model predicts remaining cycles/hours
        log_action("1", "Running ML Model for RUL Prediction")
        # Step 2: Convert RUL to Failure Window (Status determination)
        log_action("2", "Converting RUL to Failure Window / Machine Status")
        
        result = await ml_service.update_machine_from_data(machine_id, temp_path, dataset_id=dataset_id)
        
        log_action("2", "Status Update complete", f"New Status: {result['status']}, RUL: {result['predicted_rul']}")
        
        return {
            "message": f"Sensor data {file.filename} processed successfully.",
            "prediction": result
        }
    except Exception as e:
        log_action("1", "Error during ML processing", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 3. Clean up
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/model-options")
async def get_model_options():
    """Return dataset IDs and readiness metadata for UI model selection."""
    canonical_ids = ["FD001", "FD002", "FD003", "FD004"]
    try:
        available = list_available_models(ml_service.models_dir)
    except Exception as exc:
        log_error("1", f"Failed to list available models: {str(exc)}")
        available = []

    options = available

    if not options:
        options = canonical_ids

    ready_set = set(available)
    readiness = {
        dataset_id: {
            "available": dataset_id in ready_set,
            "message": "Model ready" if dataset_id in ready_set else "Model artifact not found",
        }
        for dataset_id in canonical_ids
    }

    return {
        "dataset_ids": options,
        "ready_dataset_ids": sorted(list(ready_set)),
        "readiness": readiness,
        "model_modes": get_model_mode_status(models_dir=ml_service.models_dir),
    }


@router.get("/models/options")
async def get_model_options_v2():
    """Non-conflicting alias for model selection payload."""
    return await get_model_options()
