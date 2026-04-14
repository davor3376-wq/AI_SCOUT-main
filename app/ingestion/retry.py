"""
Retry utility for Sentinel Hub (and other) API calls.
Uses exponential back-off so transient 5xx / network errors don't
permanently fail jobs.
"""
import time
import logging
import functools
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


def retry_call(fn: Callable, *args, max_attempts: int = 3, base_delay: float = 2.0, backoff: float = 2.0, **kwargs):
    """
    Inline retry helper for call sites that can't use the decorator.

    Example:
        items = list(retry_call(catalog.search, collection=col, bbox=bbox, time_interval=ti))
    """
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_attempts:
                logger.error(f"{getattr(fn, '__qualname__', fn)} failed after {max_attempts} attempts: {exc}")
                raise
            logger.warning(
                f"{getattr(fn, '__qualname__', fn)} attempt {attempt}/{max_attempts} failed "
                f"({exc}). Retrying in {delay:.1f}s…"
            )
            time.sleep(delay)
            delay *= backoff


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator: retries a function up to *max_attempts* times with
    exponential back-off on any exception in *exceptions*.

    Delays: base_delay, base_delay*backoff, base_delay*backoff^2, …
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            f"{fn.__qualname__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    logger.warning(
                        f"{fn.__qualname__} attempt {attempt}/{max_attempts} failed "
                        f"({exc}). Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator
