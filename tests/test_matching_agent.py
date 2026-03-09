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
