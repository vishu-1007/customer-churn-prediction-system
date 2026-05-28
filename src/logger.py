"""
src/logger.py
Centralized logging configuration for the entire project.
"""
import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a configured logger that writes to both console and file.

    Args:
        name: Module name (use __name__ when calling)
        log_dir: Directory to store log files

    Returns:
        logging.Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Format ───────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Console handler (INFO and above) ─────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # ── File handler (DEBUG and above) ───────────────────────
    log_file = os.path.join(log_dir, f"churn_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
