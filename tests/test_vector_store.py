import numpy as np
from embeddings.vector_store import VectorStore


def test_build_and_query():
    store = VectorStore(dimension=4)
    jobs = [
        {"id": "1", "positionName": "Python Dev", "company": "A"},
        {"id": "2", "positionName": "Java Dev", "company": "B"},
        {"id": "3", "positionName": "Data Scientist", "company": "C"},
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    store.build(jobs, embeddings)

    query = [1.0, 0.0, 0.0, 0.0]
    results = store.query(query, k=2)
    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert "vector_score" in results[0]


def test_save_and_load(tmp_path):
    store = VectorStore(dimension=4)
    jobs = [{"id": "1", "positionName": "Dev", "company": "X"}]
    embeddings = [[1.0, 0.0, 0.0, 0.0]]
    store.build(jobs, embeddings)
    store.save(str(tmp_path))

    store2 = VectorStore(dimension=4)
    store2.load(str(tmp_path))
    results = store2.query([1.0, 0.0, 0.0, 0.0], k=1)
    assert results[0]["id"] == "1"
