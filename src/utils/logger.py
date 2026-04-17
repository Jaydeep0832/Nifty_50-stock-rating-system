"""Logging setup — console + file handlers with configurable level."""

import logging
import sys
from pathlib import Path
from src.utils.config import PROJECT_ROOT


def setup_logger(name: str = "nifty50", level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Create a configured logger with console and optional file output.
    
    Args:
        name: Logger name.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file (relative to project root).
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    if log_file:
        log_path = PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
