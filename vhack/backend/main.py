from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import machines, resources, maintenance, auth
from utils.logger import log_error

app = FastAPI(title="VHACK Predictive Maintenance API")

# Global Exception Handler to avoid tracebacks in console
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log_error("SYS", f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "detail": str(exc)},
    )

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(machines.router, prefix="/api/machines", tags=["machines"])
app.include_router(resources.router, prefix="/api/resources", tags=["resources"])
app.include_router(maintenance.router, prefix="/api/maintenance", tags=["maintenance"])

@app.get("/")
async def root():
    print("Root endpoint called!")
    return {"message": "VHACK Predictive Maintenance API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
