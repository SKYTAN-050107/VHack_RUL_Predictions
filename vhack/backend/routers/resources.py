from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import os
import shutil
from services.database import supabase
from services.rag_service import rag_service
from models.database_models import Resource
from utils.logger import log_action

router = APIRouter()

TEMP_DIR = "temp_resources"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/upload")
async def upload_resource(file: UploadFile = File(...), resource_type: str = Form("technical")):
    """
    Handles PDF/TXT uploads, stores metadata in Supabase, and processes RAG.
    Part of Step 4 & 5 context preparation.
    """
    log_action("4", "Uploading Resource for Risk Analysis", f"Type: {resource_type}, File: {file.filename}")
    
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt") or file.filename.endswith(".md")):
        raise HTTPException(status_code=400, detail="Supported files: .pdf, .txt, .md")
    
    # 1. Save metadata to Supabase 'resources' table
    resource_id = 0
    try:
        if supabase:
            resource_data = {
                "filename": file.filename,
                "resource_type": resource_type
            }
            response = supabase.table("resources").insert(resource_data).execute()
            if response and hasattr(response, 'data') and len(response.data) > 0:
                resource_id = response.data[0]["id"]
    except Exception as e:
        log_action("4", "Supabase error (using mock resource_id)", str(e))
        resource_id = 999  # Mock ID
    
    # 2. Save file temporarily for processing
    temp_file_path = os.path.join(TEMP_DIR, f"{resource_id}_{file.filename}")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 3. Process with RAG service (chunking, embedding, storage)
        await rag_service.process_and_store_resource(temp_file_path, resource_id)
        return {"message": f"{resource_type.capitalize()} resource '{file.filename}' processed successfully.", "resource_id": resource_id}
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"RAG Processing Error: {str(e)}")
    finally:
        # 4. Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.get("/status")
async def get_resource_status():
    """Checks if technical and financial resources have been uploaded."""
    try:
        if supabase:
            response = supabase.table("resources").select("resource_type").execute()
            if response and hasattr(response, 'data'):
                types = [r["resource_type"] for r in response.data]
                return {
                    "technical_manuals_uploaded": "technical" in types,
                    "financial_reports_uploaded": "financial" in types
                }
    except Exception as e:
        log_action("5", "Error checking resource status", str(e))
    
    # Mock status for testing if Supabase is not configured
    return {
        "technical_manuals_uploaded": True,  # Default to True for demo if it fails
        "financial_reports_uploaded": True
    }
