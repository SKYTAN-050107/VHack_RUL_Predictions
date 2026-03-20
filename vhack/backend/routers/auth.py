from fastapi import APIRouter, HTTPException
from services.database import supabase
from models.database_models import UserAuth
from utils.logger import log_action, log_error

router = APIRouter()

@router.post("/signup")
async def signup(auth: UserAuth):
    """Sign up a new user with email and password."""
    try:
        log_action("AUTH", "Signup attempt", f"Email: {auth.email}")
        response = supabase.auth.sign_up({
            "email": auth.email,
            "password": auth.password
        })
        return {"message": "Signup successful. Check your email for confirmation.", "user": response.user}
    except Exception as e:
        log_error("AUTH", f"Signup failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
async def login(auth: UserAuth):
    """Log in a user with email and password."""
    try:
        log_action("AUTH", "Login attempt", f"Email: {auth.email}")
        response = supabase.auth.sign_in_with_password({
            "email": auth.email,
            "password": auth.password
        })
        return {
            "message": "Login successful",
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "user": response.user
        }
    except Exception as e:
        error_msg = str(e)
        log_error("AUTH", f"Login failed: {error_msg}")
        
        if "Email not confirmed" in error_msg:
            raise HTTPException(status_code=401, detail="Email not confirmed. Please check your inbox or disable 'Confirm email' in Supabase settings.")
            
        raise HTTPException(status_code=401, detail="Invalid email or password")

@router.post("/logout")
async def logout():
    """Log out the current user."""
    try:
        supabase.auth.sign_out()
        return {"message": "Logout successful"}
    except Exception as e:
        log_error("AUTH", f"Logout failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
