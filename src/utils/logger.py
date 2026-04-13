"""
Structured logger for QUEST.

- Console: human-readable coloured output.
- File: JSON-lines for machine parsing.
- Thread-safe singleton via module-level instance.

Usage:
    from src.utils.logger import get_logger
    log = get_logger("train")
    log.info("epoch started", epoch=1, lr=3e-4)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ANSI colour codes (disabled when not a tty)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stderr.isatty()

_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        colour = _COLOURS.get(level, "") if _USE_COLOUR else ""
        reset = _RESET if _USE_COLOUR else ""
        ts = time.strftime("%H:%M:%S")
        name = f"{record.name:<12}"
        msg = super().format(record)
        # Strip default prefix — we rebuild it
        msg = record.getMessage()
        return f"{colour}[{ts}] {level:<8} {name}{reset} {msg}"


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra"):
            payload.update(record.extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# QUESTLogger — thin wrapper around stdlib logging
# ---------------------------------------------------------------------------

class QUESTLogger:
    def __init__(self, name: str, log_file: Path | None = None, level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(f"quest.{name}")
        self._logger.setLevel(level)
        self._logger.propagate = False

        if not self._logger.handlers:
            # Console handler
            ch = logging.StreamHandler(sys.stderr)
            ch.setFormatter(_ColourFormatter())
            ch.setLevel(level)
            self._logger.addHandler(ch)

            # File handler (JSONL)
            if log_file is not None:
                log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
                fh.setFormatter(_JsonlFormatter())
                fh.setLevel(logging.DEBUG)
                self._logger.addHandler(fh)

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        record = self._logger.makeRecord(
            self._logger.name, level, fn="", lno=0,
            msg=self._fmt(msg, kwargs), args=(), exc_info=None,
        )
        record.extra = kwargs
        self._logger.handle(record)

    @staticmethod
    def _fmt(msg: str, kwargs: dict) -> str:
        if not kwargs:
            return msg
        parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{msg}  {parts}"

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)


# ---------------------------------------------------------------------------
# Module-level factory (memoised per name)
# ---------------------------------------------------------------------------

_registry: dict[str, QUESTLogger] = {}


def get_logger(name: str, log_file: Path | str | None = None, level: int = logging.INFO) -> QUESTLogger:
    """Return (or create) a named logger. Subsequent calls with same name return the cached instance."""
    if name not in _registry:
        _registry[name] = QUESTLogger(
            name,
            log_file=Path(log_file) if log_file else None,
            level=level,
        )
    return _registry[name]