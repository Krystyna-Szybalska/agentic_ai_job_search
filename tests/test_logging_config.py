import logging
from utils.logging_config import setup_logging


def test_setup_logging_default_is_info():
    setup_logging(verbose=False)
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_verbose_is_debug():
    setup_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_silences_urllib3():
    setup_logging(verbose=True)
    assert logging.getLogger("urllib3").level == logging.WARNING


def test_setup_logging_silences_httpx():
    setup_logging(verbose=True)
    assert logging.getLogger("httpx").level == logging.WARNING
