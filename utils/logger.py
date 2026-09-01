"""
utils/logger.py

Centralized logging configuration for the AI-DMS application.

Provides a single function, `get_logger`, that returns a configured
`logging.Logger` instance writing to both the console and a rotating
log file (path/size/level controlled entirely by config.py).
"""

import logging
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a configured logger instance.

    Args:
        name: Name of the logger, typically `__name__` of the calling module.

    Returns:
        A `logging.Logger` configured with a rotating file handler and a
        console stream handler, using settings defined in config.py.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this logger was already configured
    # (can happen if get_logger is called multiple times for the same name).
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=config.LOG_FILE_PATH,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
