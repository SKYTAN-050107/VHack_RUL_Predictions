from fastapi import APIRouter, HTTPException
from typing import List
from services.database import supabase
from models.database_models import MaintenanceLog, MaintenanceCreate, Staff
from datetime import datetime
from utils.logger import log_action

router = APIRouter()


def _resolve_machine_id_for_maintenance(requested_machine_id: int) -> int:
    """
    Ensure maintenance_logs.machine_id always points to an existing machines row.
    Replay unit IDs may not exist in machines, so create a placeholder record.
    """
    if not supabase:
        return requested_machine_id

    existing = supabase.table("machines").select("id").eq("id", requested_machine_id).limit(1).execute()
    if existing.data:
        return int(existing.data[0]["id"])

    created = supabase.table("machines").insert(
        {
            "name": f"Engine Unit {requested_machine_id}",
            "type": "ReplayUnit",
            "current_rul": 0,
            "status": "Yellow",
            "last_updated": datetime.now().isoformat(),
        }
    ).execute()

    if created.data:
        created_id = int(created.data[0]["id"])
        log_action(
            "6B",
            "Created placeholder machine for maintenance action",
            f"requested_machine_id={requested_machine_id}, created_machine_id={created_id}",
        )
        return created_id

    raise HTTPException(status_code=500, detail="Failed to resolve machine ID for maintenance request")

@router.get("/staff", response_model=List[Staff])
async def get_all_staff():
    """Fetch all technical staff members."""
    if supabase:
        response = supabase.table("staff").select("*").execute()
        return response.data
    return [
        {"id": 1, "name": "Senior Tech", "role": "Senior Technician", "specialty": "Mechanical", "status": "Available"},
        {"id": 2, "name": "Junior Mike", "role": "Junior Technician", "specialty": "Electrical", "status": "Busy"},
        {"id": 3, "name": "Expert Sarah", "role": "Maintenance Manager", "specialty": "Software", "status": "Available"}
    ]

@router.post("/staff/create")
async def create_staff(staff: Staff):
    """Add a new staff member."""
    if supabase:
        response = supabase.table("staff").insert(staff.dict(exclude={"id", "created_at"})).execute()
        return response.data[0]
    return staff

@router.delete("/staff/{staff_id}")
async def delete_staff(staff_id: int):
    """Remove a staff member."""
    if supabase:
        supabase.table("staff").delete().eq("id", staff_id).execute()
    return {"message": "Staff member removed"}

@router.post("/create")
async def create_maintenance(data: MaintenanceCreate):
    """
    Step 6A/6B: Recommendation Planning / Management Decision.
    Step 7: Generate Technical Instructions.
    """
    log_action("6B", "Management Decision Approved", f"Machine ID: {data.machine_id}, Action: {data.action_label}")
    log_action("7", "Generating Technical Instructions", f"Action: {data.action_label}")
    
    try:
        resolved_machine_id = _resolve_machine_id_for_maintenance(int(data.machine_id))
        response = supabase.table("maintenance_logs").insert({
            "machine_id": resolved_machine_id,
            "technician_name": data.technician_name,
            "action_taken": data.action_label,
            "root_cause_prediction": data.root_cause_prediction,
            "steps": data.steps,
            "components": data.components,
            "estimated_time": data.estimated_time,
            "status": "Active"
        }).execute()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create maintenance log: {str(exc)}")
    
    if not response.data:
        log_action("6B", "Error creating maintenance log")
        raise HTTPException(status_code=500, detail="Failed to create maintenance log")
    
    # Update technician status to 'Busy'
    if supabase:
        supabase.table("staff").update({"status": "Busy"}).eq("name", data.technician_name).execute()
    
    log_action("8", "Maintenance Platform updated with new task", f"Log ID: {response.data[0]['id']}")
    return response.data[0]

@router.get("/active", response_model=List[MaintenanceLog])
async def get_active_maintenance():
    """Step 8: Active monitoring dashboard."""
    log_action("8", "Fetching Active Maintenance Tasks")
    response = supabase.table("maintenance_logs").select("*").eq("status", "Active").execute()
    return response.data

@router.post("/complete/{maintenance_id}")
async def complete_maintenance(maintenance_id: int, action_taken: str, root_cause: str):
    """
    Step 9: Technician Report.
    Step 10: Technician Feedback Loop.
    """
    log_action("9", "Technician Report submitted", f"Log ID: {maintenance_id}")
    log_action("10", "Technician Feedback Loop initiated", f"Recording root cause: {root_cause}")
    
    response = supabase.table("maintenance_logs").update({
        "status": "Completed",
        "action_taken": action_taken,
        "root_cause": root_cause,
        "completion_date": datetime.now().isoformat()
    }).eq("id", maintenance_id).execute()
    
    if not response.data:
        log_action("10", "Error updating maintenance log", f"Log ID: {maintenance_id}")
        raise HTTPException(status_code=404, detail="Maintenance log not found")
        
    log_id = response.data[0]['id']
    machine_id = response.data[0]['machine_id']
    tech_name = response.data[0]['technician_name']

    # Final Step: Update Machine Status to Green and Reset RUL
    if supabase:
        supabase.table("machines").update({
            "status": "Green",
            "current_rul": 500, # Reset to baseline
            "last_updated": datetime.now().isoformat()
        }).eq("id", machine_id).execute()
        
        # Free up the technician
        supabase.table("staff").update({"status": "Available"}).eq("name", tech_name).execute()

    log_action("10", f"Feedback Loop complete. Machine {machine_id} reset to healthy state.")
    return {"message": f"Maintenance job {log_id} completed successfully. Machine RUL reset."}
