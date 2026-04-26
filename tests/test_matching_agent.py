import logging
from unittest.mock import MagicMock
from agents.matching_agent import analyze_job


def test_analyze_job_returns_dict():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"score": 7, "matching_skills": ["Python"], "missing_skills": ["Java"], "summary": "Good match."}'

    job = {
        "id": "1",
        "positionName": "Data Scientist",
        "company": "Acme",
        "location": "Remote",
        "salary": "100k",
        "description": "Python and ML experience required.",
        "vector_score": 0.85,
    }
    result = analyze_job("Experienced Python developer", job, mock_llm)

    assert result["id"] == "1"
    assert result["llm_score"] == 7
    assert "matching_skills" in result
    assert "summary" in result


def test_analyze_job_handles_invalid_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I cannot parse this"

    job = {"id": "2", "positionName": "Dev", "company": "X",
           "location": "NY", "salary": "80k", "description": "Test", "vector_score": 0.5}
    result = analyze_job("Python dev", job, mock_llm)

    assert result["id"] == "2"
    assert result["llm_score"] == 0  # fallback


def test_analyze_job_logs_prompt_and_response_at_debug(caplog):
    cv = "Python developer"
    job = {"positionName": "Dev", "company": "Acme", "location": "Remote",
           "salary": "N/A", "description": "Role", "id": "1"}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"score": 7, "matching_skills": [], "missing_skills": [], "summary": "ok"}'

    with caplog.at_level(logging.DEBUG, logger="agents.matching_agent"):
        analyze_job(cv, job, mock_llm)

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    prompt_logs = [m for m in debug_messages if "Prompt to LLM" in m]
    response_logs = [m for m in debug_messages if "Raw LLM response" in m]
    assert len(prompt_logs) == 1
    assert len(response_logs) == 1
    assert "job=1" in prompt_logs[0]
    assert "Python developer" in prompt_logs[0]
    assert "ok" in response_logs[0]
