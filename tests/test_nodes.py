from unittest.mock import patch, MagicMock
from graph.state import JobSearchState


def _make_state(**overrides) -> JobSearchState:
    defaults = {
        "cv_path": "", "cv_text": "", "cv_embedding": [],
        "retrieved_jobs": [], "matched_jobs": [], "critic_feedback": "",
        "validated_jobs": [], "report_path": "", "remotive_search": "",
    }
    defaults.update(overrides)
    return JobSearchState(**defaults)


def test_parse_cv_node():
    from graph.nodes import parse_cv
    state = _make_state(cv_path="nonexistent.pdf")
    with patch("graph.nodes.extract_text_from_pdf", return_value="John Doe, Python developer"):
        result = parse_cv(state)
    assert result["cv_text"] == "John Doe, Python developer"


def test_embed_cv_node():
    from graph.nodes import embed_cv
    state = _make_state(cv_text="Python developer with 5 years experience")
    with patch("graph.nodes.get_embedder") as mock_get, \
         patch("graph.nodes.embed_text", return_value=[0.1] * 384):
        result = embed_cv(state)
    assert len(result["cv_embedding"]) == 384


def test_retrieve_jobs_node():
    from graph.nodes import retrieve_jobs
    state = _make_state(cv_embedding=[0.1] * 384)
    with patch("graph.nodes.VectorStore") as MockStore:
        instance = MockStore.return_value
        instance.query.return_value = [{"id": "1", "positionName": "Dev", "vector_score": 0.9}]
        instance.load.return_value = None
        result = retrieve_jobs(state)
    assert len(result["retrieved_jobs"]) == 1


def test_fetch_remote_jobs_node():
    from graph.nodes import fetch_remote_jobs
    state = _make_state(cv_text="Python ML engineer")
    fake_jobs = [
        {"id": "1", "positionName": "ML Engineer", "company": "Co", "location": "Remote",
         "salary": "N/A", "description": "Build ML stuff", "url": "https://example.com/1"},
    ]
    with patch("graph.nodes.fetch_remotive_jobs", return_value=fake_jobs):
        result = fetch_remote_jobs(state)
    assert len(result["retrieved_jobs"]) == 1
    assert result["retrieved_jobs"][0]["positionName"] == "ML Engineer"


def test_node_emits_decorator_and_logger_messages(caplog):
    import logging
    from graph.nodes import retrieve_jobs
    state = _make_state(cv_embedding=[0.1] * 384)
    with patch("graph.nodes.VectorStore") as MockStore:
        MockStore.return_value.query.return_value = [{"id": "1"}]
        MockStore.return_value.load.return_value = None
        with caplog.at_level(logging.INFO, logger="graph.nodes"):
            retrieve_jobs(state)
    messages = [r.message for r in caplog.records]
    assert any("Retrieve jobs" in m and "started" in m for m in messages)
    assert any("Retrieve jobs" in m and "finished" in m for m in messages)
    assert any("Retrieved 1 jobs" in m for m in messages)


def test_fetch_remote_jobs_uses_state_search():
    from graph.nodes import fetch_remote_jobs
    state = _make_state(remotive_search="ML engineer")
    with patch("graph.nodes.fetch_remotive_jobs", return_value=[]) as mock_fetch:
        fetch_remote_jobs(state)
    mock_fetch.assert_called_once()
    args, kwargs = mock_fetch.call_args
    # First positional arg or 'search' kwarg should be "ML engineer"
    assert args[0] == "ML engineer" or kwargs.get("search") == "ML engineer"
