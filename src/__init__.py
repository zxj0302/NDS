from .baselines import *
from .painter import *
from loguru import logger
import sys
import json

def set_logger():
    # Remove default configuration
    logger.remove()
    
    # Console output - with background colors and bold text
    logger.add(
        sys.stderr,
        format=(
            "<bg #2C3E50><white><bold>{time:YYYY-MM-DD HH:mm:ss}</bold></white></bg #2C3E50> "
            "| <level><bold>{level: <8}</bold></level> | "
            "<cyan><bold>{name}:{function}:{line}</bold></cyan> | "
            "<level><bold>{message}</bold></level>"
        ),
        colorize=True,
        level="DEBUG"
    )
    
    # File output - detailed logs (no colors)
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        level="DEBUG"
    )
    
    # Separate error log file
    logger.add(
        "logs/error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        level="ERROR"
    )
    
    # Configure different level colors (Loguru built-in configuration)
    # DEBUG: Blue background (lowest severity)
    # INFO: Green background (safe)
    # SUCCESS: Cyan background (successful)
    # WARNING: Yellow background (moderate severity)
    # ERROR: Orange background (high severity)
    # CRITICAL: Red background (extreme severity)
    
    logger.level("DEBUG", color="<bg #3498DB><white>")      # Blue background
    logger.level("INFO", color="<bg #2ECC71><white>")       # Green background
    logger.level("SUCCESS", color="<bg #1ABC9C><white>")    # Cyan background
    logger.level("WARNING", color="<bg #F39C12><black>")    # Yellow background
    logger.level("ERROR", color="<bg #E67E22><white>")      # Orange background
    logger.level("CRITICAL", color="<bg #C0392B><white>")   # Red background
    
    logger.info("Logger configured successfully")
    
    return logger

set_logger()