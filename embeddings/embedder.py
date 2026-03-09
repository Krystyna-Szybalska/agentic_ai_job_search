from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL


def get_embedder() -> SentenceTransformer:
    """Load and return the sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str, embedder: SentenceTransformer) -> list[float]:
    """Embed a single text string."""
    return embedder.encode(text).tolist()


def embed_batch(texts: list[str], embedder: SentenceTransformer) -> list[list[float]]:
    """Embed a list of text strings."""
    return embedder.encode(texts).tolist()
