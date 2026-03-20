# Copilot Instructions for VHACK

## Project Scope
- This repository contains a predictive maintenance platform with:
- FastAPI backend in backend/
- Streamlit frontend in frontend/
- Root Streamlit multipage app using app.py and pages/

## Quick Start Commands
- Backend:
  - cd backend
  - pip install -r requirements.txt
  - python main.py
- Frontend:
  - cd frontend
  - pip install -r requirements.txt
  - streamlit run app.py
- Root Streamlit app:
  - pip install -r requirements.txt
  - streamlit run app.py

## Environment Requirements
- Configure backend environment variables before running API features:
  - SUPABASE_URL
  - SUPABASE_KEY
  - GOOGLE_API_KEY
- Backend serves on port 8000 by default.
- Streamlit serves on port 8501 by default.

## Architecture Boundaries
- backend/main.py wires routers and CORS; keep route registration here.
- backend/routers/ contains API endpoint handlers by domain:
  - auth.py, machines.py, maintenance.py, resources.py
- backend/services/ contains business logic and integrations:
  - database.py for Supabase client access
  - ml_service.py for RUL/status logic
  - rag_service.py for document chunking and embeddings
  - reasoning_service.py for LLM-based reasoning
- backend/models/database_models.py contains Pydantic request/response models.
- frontend/app.py handles Streamlit authentication flow against backend API.
- frontend/pages/ contains frontend-specific pages.
- pages/ contains root Streamlit multipage screens used by root app.py.

## Coding Conventions in This Repo
- Keep FastAPI endpoints async when possible.
- Keep API contracts in Pydantic models under backend/models/.
- Keep integration/business logic in backend/services/, not inline inside routers.
- Reuse logging helpers from backend/utils/logger.py for backend actions/errors.
- Maintain current API prefix conventions:
  - /api/auth
  - /api/machines
  - /api/resources
  - /api/maintenance

## Known Pitfalls
- backend/services/ml_service.py currently contains placeholder logic; avoid claiming production-grade RUL model behavior unless model integration is added.
- backend/services/rag_service.py includes placeholder document loading behavior and should be treated as an integration scaffold.
- If GOOGLE_API_KEY is not set, reasoning flows may degrade to fallback behavior.
- CORS is currently permissive in backend/main.py; tighten allow_origins for production.

## Where to Look First
- Product/problem context: stuff.md
- API entrypoint and middleware: backend/main.py
- Data schema setup: backend/init_supabase.sql
- API data models: backend/models/database_models.py
- Frontend auth/API usage: frontend/app.py

## Link, Do Not Duplicate
- When implementing features tied to business flow or case-study context, reference existing source context in stuff.md instead of duplicating long narrative text in code comments.
- For DB behavior, align with backend/init_supabase.sql and avoid redefining schema assumptions in multiple places.

## Agent Working Rules for This Workspace
- Prefer small, targeted edits over broad refactors.
- Preserve existing endpoint shapes and response fields unless explicitly asked to change contracts.
- When adding new dependencies, update the nearest requirements.txt used by that runtime (root, backend, or frontend).
- If both root pages/ and frontend/pages/ are touched, clearly keep responsibilities separated.

## Validation Expectations After Changes
- For backend changes, run from backend/:
  - python main.py
- For frontend changes, run from frontend/:
  - streamlit run app.py
- For root dashboard changes, run from repo root:
  - streamlit run app.py