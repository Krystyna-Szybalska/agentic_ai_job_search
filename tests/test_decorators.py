import logging
import pytest
from utils.decorators import log_step


def test_log_step_logs_start_and_finish(caplog):
    @log_step("My step")
    def my_func():
        return "result"

    with caplog.at_level(logging.INFO):
        result = my_func()

    assert result == "result"
    messages = [r.message for r in caplog.records]
    assert any("My step started" in m for m in messages)
    assert any("My step finished" in m for m in messages)


def test_log_step_propagates_return_value():
    @log_step("Echo")
    def echo(x):
        return x * 2

    assert echo(21) == 42


def test_log_step_logs_exception_and_reraises(caplog):
    @log_step("Boom")
    def bad():
        raise ValueError("kaboom")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError, match="kaboom"):
            bad()

    assert any("Boom failed" in r.message for r in caplog.records)


def test_log_step_uses_function_name_when_no_name_given(caplog):
    @log_step()
    def my_named_func():
        return None

    with caplog.at_level(logging.INFO):
        my_named_func()

    assert any("my_named_func started" in r.message for r in caplog.records)


def test_log_step_logger_uses_caller_module(caplog):
    @log_step("Test")
    def f():
        return None

    with caplog.at_level(logging.INFO):
        f()

    # logger name should be the test module's name, not 'utils.decorators'
    assert any(r.name == __name__ for r in caplog.records)
