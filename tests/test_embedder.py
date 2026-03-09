from embeddings.embedder import get_embedder, embed_text, embed_batch


def test_embed_text_returns_list():
    embedder = get_embedder()
    result = embed_text("Software engineer with Python experience", embedder)
    assert isinstance(result, list)
    assert len(result) == 384  # all-MiniLM-L6-v2 dimension


def test_embed_batch_returns_list_of_lists():
    embedder = get_embedder()
    texts = ["Python developer", "Data scientist", "Product manager"]
    result = embed_batch(texts, embedder)
    assert len(result) == 3
    assert len(result[0]) == 384
