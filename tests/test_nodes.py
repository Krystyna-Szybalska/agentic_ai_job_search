from unittest.mock import patch, MagicMock
from graph.state import JobSearchState


def test_parse_cv_node(tmp_path):
    from graph.nodes import parse_cv
    state = JobSearchState(
        cv_path="nonexistent.pdf", cv_text="", cv_embedding=[],
        retrieved_jobs=[], matched_jobs=[], critic_feedback="",
        validated_jobs=[], human_approved=False, report_path=""
    )
    with patch("graph.nodes.extract_text_from_pdf", return_value="John Doe, Python developer"):
        result = parse_cv(state)
    assert result["cv_text"] == "John Doe, Python developer"


def test_embed_cv_node():
    from graph.nodes import embed_cv
    state = JobSearchState(
        cv_path="", cv_text="Python developer with 5 years experience",
        cv_embedding=[], retrieved_jobs=[], matched_jobs=[],
        critic_feedback="", validated_jobs=[], human_approved=False, report_path=""
    )
    with patch("graph.nodes.get_embedder") as mock_get, \
         patch("graph.nodes.embed_text", return_value=[0.1] * 384):
        result = embed_cv(state)
    assert len(result["cv_embedding"]) == 384


def test_retrieve_jobs_node():
    from graph.nodes import retrieve_jobs
    state = JobSearchState(
        cv_path="", cv_text="", cv_embedding=[0.1] * 384,
        retrieved_jobs=[], matched_jobs=[], critic_feedback="",
        validated_jobs=[], human_approved=False, report_path=""
    )
    with patch("graph.nodes.VectorStore") as MockStore:
        instance = MockStore.return_value
        instance.query.return_value = [{"id": "1", "positionName": "Dev", "vector_score": 0.9}]
        instance.load.return_value = None
        result = retrieve_jobs(state)
    assert len(result["retrieved_jobs"]) == 1
