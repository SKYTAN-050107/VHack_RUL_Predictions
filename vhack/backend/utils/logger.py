import logging
import sys

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"

class ColorFormatter(logging.Formatter):
    def format(self, record):
        step = getattr(record, 'step', 'N/A')
        
        # Color based on step category
        step_color = CYAN
        if step in ["1", "2"]: step_color = GREEN  # ML Steps
        elif step in ["3", "4", "5"]: step_color = MAGENTA # AI Steps
        elif step in ["6A", "6B", "7"]: step_color = YELLOW # Decision Steps
        elif step in ["8", "9", "10"]: step_color = BOLD # Platform Steps

        level_color = RESET
        if record.levelno >= logging.ERROR: level_color = RED
        elif record.levelno >= logging.WARNING: level_color = YELLOW

        log_fmt = f"{BOLD}[Step {step}]{RESET} {step_color}%(message)s{RESET}"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure logger
logger = logging.getLogger("VHACK-PM")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter())
logger.addHandler(handler)

def log_action(step: str, action: str, details: str = ""):
    """
    Logs an action with color coding and step grouping.
    """
    msg = f"{BOLD}{action}{RESET}"
    if details:
        msg += f" -> {details}"
    logger.info(msg, extra={'step': step})

def log_error(step: str, error_msg: str):
    """
    Logs an error in red without traceback.
    """
    logger.error(f"{RED}{BOLD}ERROR:{RESET} {error_msg}", extra={'step': step})
