import os
from dotenv import load_dotenv

load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "resources")

# Gemini/LLM Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# App Settings
APP_NAME = "VHACK-PM-Backend"
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
