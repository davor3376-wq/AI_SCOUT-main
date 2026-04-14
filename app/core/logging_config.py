"""
Structured logging configuration.

Set LOG_FORMAT=json  (default) for machine-parseable output compatible with
Datadog, CloudWatch, Grafana Loki, etc.

Set LOG_FORMAT=text for human-friendly output during local development.
Set LOG_LEVEL=DEBUG|INFO|WARNING|ERROR (default INFO).
"""

import logging
import os
import sys


def setup_logging() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "json").lower()

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        try:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
                rename_fields={"asctime": "timestamp", "levelname": "level"},
            )
        except ImportError:
            # python-json-logger not installed — fall back to text silently
            formatter = logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
    else:
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "rasterio", "sentinelhub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
