import logging
from unittest.mock import MagicMock

from agents.critic_agent import critique_matches


def test_critique_returns_reordered_jobs():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"verdict": "approved", "feedback": "Ranking looks correct.", "suggested_ranking": ["2", "1", "3"]}'

    jobs = [
        {"id": "1", "positionName": "Dev A", "company": "X", "llm_score": 8},
        {"id": "2", "positionName": "Dev B", "company": "Y", "llm_score": 9},
        {"id": "3", "positionName": "Dev C", "company": "Z", "llm_score": 7},
    ]
    validated, feedback = critique_matches("Python dev CV", jobs, mock_llm)

    assert validated[0]["id"] == "2"
    assert validated[1]["id"] == "1"
    assert "Ranking looks correct" in feedback


def test_critique_fallback_on_invalid_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "cannot parse"

    jobs = [{"id": "1", "positionName": "Dev", "company": "X", "llm_score": 8}]
    validated, feedback = critique_matches("CV text", jobs, mock_llm)

    assert len(validated) == 1  # returns original order
    assert isinstance(feedback, str)


def test_critique_matches_logs_prompt_and_response_at_debug(caplog):
    cv = "Python developer"
    matched = [{"id": "1", "positionName": "Dev", "company": "Acme", "llm_score": 7}]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"verdict": "approved", "feedback": "ok feedback", "suggested_ranking": ["1"]}'

    with caplog.at_level(logging.DEBUG, logger="agents.critic_agent"):
        critique_matches(cv, matched, mock_llm)

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    prompt_logs = [m for m in debug_messages if "Prompt to LLM" in m]
    response_logs = [m for m in debug_messages if "Raw LLM response" in m]
    assert len(prompt_logs) == 1
    assert len(response_logs) == 1
    assert "Python developer" in prompt_logs[0]
    assert "ok feedback" in response_logs[0]
