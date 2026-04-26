import logging

import pytest
from unittest.mock import patch, Mock, MagicMock


SAMPLE_API_RESPONSE = {
    "jobs": [
        {
            "id": 12345,
            "url": "https://remotive.com/remote-jobs/software-dev/senior-python-12345",
            "title": "Senior Python Developer",
            "company_name": "Acme Corp",
            "category": "Software Development",
            "tags": ["python", "django"],
            "job_type": "full_time",
            "publication_date": "2026-04-01T00:00:00",
            "candidate_required_location": "Worldwide",
            "salary": "$100k - $150k",
            "description": "<p>We are looking for a <strong>Python developer</strong>.</p><ul><li>Django</li><li>REST APIs</li></ul>",
        },
        {
            "id": 67890,
            "url": "https://remotive.com/remote-jobs/software-dev/ml-engineer-67890",
            "title": "ML Engineer",
            "company_name": "DataCo",
            "category": "Software Development",
            "tags": ["python", "ml"],
            "job_type": "full_time",
            "publication_date": "2026-04-02T00:00:00",
            "candidate_required_location": "Europe",
            "salary": "",
            "description": "<div>Build ML pipelines.</div>",
        },
    ]
}


class TestNormalizeRemotiveJob:
    def test_maps_fields_to_internal_format(self):
        from tools.remotive_client import normalize_remotive_job

        raw = SAMPLE_API_RESPONSE["jobs"][0]
        result = normalize_remotive_job(raw)

        assert result["positionName"] == "Senior Python Developer"
        assert result["company"] == "Acme Corp"
        assert result["location"] == "Worldwide"
        assert result["salary"] == "$100k - $150k"
        assert result["id"] == "12345"
        assert result["url"] == raw["url"]

    def test_strips_html_from_description(self):
        from tools.remotive_client import normalize_remotive_job

        raw = SAMPLE_API_RESPONSE["jobs"][0]
        result = normalize_remotive_job(raw)

        assert "<p>" not in result["description"]
        assert "<strong>" not in result["description"]
        assert "<ul>" not in result["description"]
        assert "Python developer" in result["description"]

    def test_handles_empty_salary(self):
        from tools.remotive_client import normalize_remotive_job

        raw = SAMPLE_API_RESPONSE["jobs"][1]
        result = normalize_remotive_job(raw)

        assert result["salary"] == "N/A"


class TestFetchRemotiveJobs:
    @patch("tools.remotive_client.requests.get")
    def test_returns_normalized_jobs(self, mock_get):
        from tools.remotive_client import fetch_remotive_jobs

        mock_response = Mock()
        mock_response.json.return_value = SAMPLE_API_RESPONSE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        results = fetch_remotive_jobs("python developer", limit=10)

        assert len(results) == 2
        assert results[0]["positionName"] == "Senior Python Developer"
        assert results[1]["positionName"] == "ML Engineer"
        mock_get.assert_called_once()

    @patch("tools.remotive_client.requests.get")
    def test_raises_on_http_error(self, mock_get):
        from tools.remotive_client import fetch_remotive_jobs

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("503 Service Unavailable")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="503"):
            fetch_remotive_jobs("python", limit=5)


def test_fetch_remotive_jobs_logs_request_at_debug(caplog):
    from tools.remotive_client import fetch_remotive_jobs

    mock_response = MagicMock()
    mock_response.json.return_value = {"jobs": []}
    mock_response.raise_for_status = MagicMock()

    with patch("tools.remotive_client.requests.get", return_value=mock_response):
        with caplog.at_level(logging.DEBUG, logger="tools.remotive_client"):
            fetch_remotive_jobs("python", limit=5)

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    request_logs = [m for m in debug_messages if "GET" in m or "remotive" in m.lower()]
    assert len(request_logs) >= 1
    assert "remotive.com/api/remote-jobs" in request_logs[0]
    assert "python" in request_logs[0]
    assert "5" in request_logs[0]
