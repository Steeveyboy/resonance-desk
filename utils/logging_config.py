"""Logging setup shared by every page.

Streamlit re-executes the page script on every interaction and may re-import
edited modules, so :func:`setup_logging` is idempotent: it marks the loggers
it configures (logger objects outlive module reloads) and skips them after.

Configure with ``LOG_LEVEL`` (default ``INFO``) and, optionally, ``LOG_FILE``
to also write logs to disk.
"""
from __future__ import annotations

import logging

from utils.config import settings

#: Top-level namespaces owned by this app. Handlers attach here rather than on
#: the root logger, so Streamlit's, httpx's and SQLAlchemy's loggers keep their
#: own configuration.
_APP_LOGGERS = ("resonance", "agents", "simulation", "utils")

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"

_MARKER = "_resonance_configured"


def setup_logging() -> None:
    """Attach console (and optional file) handlers to the app's loggers."""
    if getattr(logging.getLogger(_APP_LOGGERS[0]), _MARKER, False):
        return

    level = logging.getLevelName((settings.log_level or "INFO").upper())
    if not isinstance(level, int):
        level = logging.INFO

    formatter = logging.Formatter(_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if settings.log_file:
        handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)

    for name in _APP_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        for handler in handlers:
            logger.addHandler(handler)
        setattr(logger, _MARKER, True)
