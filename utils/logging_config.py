"""Central logging configuration for the job search pipeline."""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger. Called once from main.py at startup.

    verbose=True sets DEBUG level (LLM prompts, raw responses, HTTP details).
    verbose=False sets INFO level (node entry/exit, key counts).
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
