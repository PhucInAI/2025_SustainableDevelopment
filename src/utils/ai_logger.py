# ai_logger.py
"""
Lightweight, color-friendly logging for console + optional rotating file.

Usage:
    from ai_logger import aiLogger, get_logger, configure_logger

    aiLogger.info("Hello")  # default global logger
    log = get_logger("my.module")
    log.debug("details...")

Env overrides:
    AI_LOG_LEVEL=INFO
    AI_LOG_FILE=/path/to/app.log
    AI_LOG_NO_COLOR=1
"""

# pylint: disable=W0718, W0602, W0603

from __future__ import annotations

import os
import sys
from typing import Optional
import logging
import logging.handlers
import colorama

__CONFIGURED = False

# ########################################################################
# Color support
# ########################################################################


_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"

ANSI = {
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",

    "light_red": "\x1b[91m",
    "light_green": "\x1b[92m",
    "light_yellow": "\x1b[93m",
    "light_blue": "\x1b[94m",
    "light_magenta": "\x1b[95m",
    "light_cyan": "\x1b[96m",
    "light_white": "\x1b[97m",
}


def _stream_supports_color(stream: object) -> bool:
    """Return True if stream is a TTY and likely supports ANSI color."""
    if not hasattr(stream, "isatty"):
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _maybe_enable_colorama():
    """Initialize colorama on Windows if available (no hard dependency)."""
    try:
        colorama.just_fix_windows_console()
        return True
    except Exception:
        return False


# ########################################################################
# Formatter
# ########################################################################


class ColorFormatter(logging.Formatter):
    """
    Colorized formatter that highlights the level and message.
    Falls back to plain text when color is disabled or not supported.
    """

    DEFAULT_FMT = (
        "Process:%(process)d||%(asctime)s-%(levelname)s "
        "[%(filename)s:%(funcName)s():%(lineno)d]: %(message)s"
    )
    DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

    LEVEL_STYLE = {
        logging.DEBUG: ANSI["cyan"],
        logging.INFO: ANSI["light_green"],
        logging.WARNING: ANSI["yellow"],
        logging.ERROR: ANSI["red"],
        logging.CRITICAL: _BOLD + ANSI["red"],
    }

    def __init__(
                self,
                fmt: Optional[str] = None,
                datefmt: Optional[str] = None,
                use_color: Optional[bool] = None,
                stream: Optional[object] = None
                ):
        super().__init__(fmt or self.DEFAULT_FMT, datefmt or self.DEFAULT_DATEFMT)
        # Decide coloring:
        if use_color is None:
            # auto: color only for TTY if supported
            use_color = _stream_supports_color(stream or sys.stdout)
            if sys.platform.startswith("win"):
                # Try enabling Windows ANSI support
                use_color = _maybe_enable_colorama() and use_color
        self.use_color = bool(use_color)

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            style = self.LEVEL_STYLE.get(record.levelno, "")
            # Colorize levelname and message; keep the rest plain
            original_levelname = record.levelname
            original_msg = record.getMessage()

            record.levelname = f"{style}{original_levelname}{_RESET}"
            record.msg = f"{style}{original_msg}{_RESET}"

            try:
                return super().format(record)
            finally:
                # Restore
                record.levelname = original_levelname
                record.msg = original_msg
        else:
            return super().format(record)


# ########################################################################
# Configuration
# ########################################################################


def configure_logger(
                    name: str = "aiLog",
                    level: Optional[int | str] = None,
                    use_color: Optional[bool] = None,
                    log_to_file: bool = False,
                    filename: Optional[str] = None,
                    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                    backup_count: int = 5,
                ) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name: Logger name.
        level: Logging level or name. Defaults to env AI_LOG_LEVEL or DEBUG.
        use_color: Force color on/off. Defaults to auto-detect.
        log_to_file: If True, adds a RotatingFileHandler.
        filename: File path for rotating file handler. Defaults to env AI_LOG_FILE.
        max_bytes: Rotation size.
        backup_count: Number of rotated backups to keep.
    """
    global __CONFIGURED

    # --------------------------------------------------------------------
    # Env-driven defaults
    # --------------------------------------------------------------------
    env_level = os.getenv("AI_LOG_LEVEL")
    env_file = os.getenv("AI_LOG_FILE")
    env_no_color = os.getenv("AI_LOG_NO_COLOR")

    if level is None:
        level = env_level or "DEBUG"
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper())
        if not isinstance(level, int):
            level = logging.DEBUG

    if filename is None:
        filename = env_file

    if use_color is None and env_no_color:
        use_color = False

    logger = logging.getLogger(name)
    logger.setLevel(level)


    # --------------------------------------------------------------------
    # Clear existing handlers to avoid duplicates
    # --------------------------------------------------------------------
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)


    # --------------------------------------------------------------------
    # Console handler
    # --------------------------------------------------------------------
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(ColorFormatter(use_color=use_color, stream=sys.stdout))
    logger.addHandler(console)

    # --------------------------------------------------------------------
    # Optional rotating file handler (plain, no color)
    # --------------------------------------------------------------------
    if log_to_file or filename:
        filename = filename or "ai.log"
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except Exception:
            # If no directory part or failed to create, ignore and let handler throw if needed
            pass

        file_fmt = "Process:%(process)d||%(asctime)s-%(levelname)s [%(filename)s:%(funcName)s():%(lineno)d]: %(message)s"
        file_datefmt = "%Y-%m-%d %H:%M:%S"
        file_handler = logging.handlers.RotatingFileHandler(
                                                            filename,
                                                            maxBytes=max_bytes,
                                                            backupCount=backup_count,
                                                            encoding="utf-8"
                                                            )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(file_fmt, datefmt=file_datefmt))
        logger.addHandler(file_handler)

    # Prevent double logging via root
    logger.propagate = False

    __CONFIGURED = True

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger; configures default logger once if needed.
    """
    global __CONFIGURED
    base_name = "aiLog"
    if not __CONFIGURED:
        configure_logger(base_name)
    return logging.getLogger(base_name if not name else f"{base_name}.{name}")


# Backwards-compatible global
aiLogger = get_logger()
