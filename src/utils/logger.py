"""
Structured logging for QUEST.
Uses Python's logging module with rich formatting for the console
and plain text for file logs — no external log server needed.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str | None = None, level: str = "INFO") -> logging.Logger:
    """
    Create and return a logger with console + optional file handlers.

    Args:
        name:    Logger name (usually __name__ of the calling module).
        log_dir: If provided, also writes to a timestamped file in this dir.
        level:   Logging level string — "DEBUG", "INFO", "WARNING", "ERROR".

    Returns:
        Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__, log_dir="results/logs")
        >>> logger.info("Starting training")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(log_dir) / f"{name.replace('.', '_')}_{timestamp}.log"
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
