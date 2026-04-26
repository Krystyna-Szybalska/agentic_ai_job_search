"""Reusable function decorators."""

import functools
import logging
import time


def log_step(name: str | None = None):
    """Decorator that logs entry/exit and timing of a function.

    The logger is resolved from the wrapped function's module so messages
    are attributed to the calling module (e.g. 'graph.nodes'), not 'utils.decorators'.
    Exceptions are logged with stack trace via logger.exception() and re-raised.
    """
    def decorator(func):
        step_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.info("-> %s started", step_name)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[OK] %s finished (%.2fs)", step_name, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                logger.exception("[FAIL] %s failed (%.2fs)", step_name, elapsed)
                raise
        return wrapper
    return decorator
