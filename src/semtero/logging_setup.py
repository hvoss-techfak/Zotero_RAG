"""Logging configuration helpers.

Goal: keep console output readable during long background embedding runs.

- Suppress noisy HTTP client per-request logs (httpx/httpcore/urllib3).
- Optionally suppress overly chatty third-party SDKs.

Call :func:`setup_logging` early in the entrypoint.
"""

from __future__ import annotations

import logging
import os


_NOISY_LOGGERS: tuple[str, ...] = (
    # Common HTTP client libraries
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    # Ollama python client can be chatty depending on version
    "ollama",
)

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def parse_log_level(value: str | int | None, default: int = logging.WARNING) -> int:
    """Parse an env/config log level into a logging module constant."""

    if isinstance(value, int):
        return value

    if value is None:
        return default

    normalized = str(value).strip().upper()
    if not normalized:
        return default

    if normalized.isdigit():
        return int(normalized)

    return getattr(logging, normalized, default)


def setup_logging(
    *,
    level: str | int | None = None,
    quiet_http: bool = True,
    quiet_http_level: str | int | None = None,
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """Apply opinionated logging defaults for the whole application."""

    root_level = parse_log_level(level or os.getenv("LOG_LEVEL"), logging.WARNING)
    noisy_level = parse_log_level(
        quiet_http_level or os.getenv("NOISY_LOG_LEVEL"),
        logging.WARNING,
    )

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=root_level, format=fmt)
    else:
        root_logger.setLevel(root_level)
        for handler in root_logger.handlers:
            handler.setLevel(root_level)
            handler.setFormatter(logging.Formatter(fmt))

    if quiet_http:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(noisy_level)

    # Keep framework internals quiet unless explicitly raised.
    logging.getLogger("mcp").setLevel(noisy_level)
    logging.getLogger("anyio").setLevel(noisy_level)
