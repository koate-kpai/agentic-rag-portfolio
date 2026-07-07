# logger.py — Structured JSON logger for app2-documind
#
# ARCHITECTURAL DECISION: Environment-driven log level
# -----------------------------------------------------
# Twelve-factor app principle #3 (Store config in the environment) demands that
# operational concerns like verbosity be externalized.  Hard-coding DEBUG means
# every production request dumps internal LLM message traces, tool call payloads,
# and timing data — noise that obscures real incidents and increases Cloud
# Logging costs.  Conversely, hard-coding INFO means developers can't easily
# debug locally without a code change.
#
# We default to INFO (the standard production level) and let developers
# override via `LOG_LEVEL=DEBUG` in their local .env file.  This way:
#   - Production containers get INFO without any deploy-time configuration.
#   - Developers get DEBUG automatically when they're working locally.
#   - On-call engineers can bump verbosity via `gcloud run deploy --set-env-vars`
#     without a new build, because Cloud Run supports live env var updates.
#   - CI pipelines can set `LOG_LEVEL=WARNING` to reduce noise in test output.

import logging
import os
import sys


def setup_logger(name: str = "documind") -> logging.Logger:
    """
    Returns a configured logger instance.

    The log level is read from the LOG_LEVEL environment variable and defaults
    to INFO when not set.  Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    # Read verbosity from the environment, falling back to INFO for production.
    # os.getenv returns None if the variable is absent, so we fall through to
    # "INFO" — this is the standard production level that avoids leaking
    # internal decision traces (LLM prompts, tool arguments, etc.).
    level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(level_name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Structured JSON output so Cloud Logging can index fields individually.
        # The "message" field holds the log text while structured fields (time,
        # level, name) enable dashboard-level filtering without text parsing.
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
