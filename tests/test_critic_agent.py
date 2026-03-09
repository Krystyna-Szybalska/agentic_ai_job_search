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
