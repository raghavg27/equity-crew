"""
Centralized logging configuration for the AI-Powered Stocks Analyser.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
from datetime import datetime


def setup_logging() -> None:
    """
    Configure the root logger for the project.
    - Console: INFO level with a clean, human-readable format.
    - File:    DEBUG level with full detail (timestamps, module, line numbers).
               Written to logs/analyser_YYYYMMDD.log
    """
    os.makedirs("logs", exist_ok=True)

    root_logger = logging.getLogger("stocks_analyser")
    root_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if setup_logging() is called more than once
    if root_logger.handlers:
        return

    # ── Console Handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    # ── File Handler ───────────────────────────────────────────────────────────
    log_filename = os.path.join(
        "logs", f"analyser_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s:%(funcName)s:%(lineno)d — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'stocks_analyser' namespace.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    setup_logging()  # Idempotent — safe to call multiple times
    return logging.getLogger(f"stocks_analyser.{name}")
