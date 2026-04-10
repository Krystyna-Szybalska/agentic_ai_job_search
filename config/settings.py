import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
TOP_K = int(os.getenv("TOP_K", "10"))
TOP_N_FOR_CRITIC = int(os.getenv("TOP_N_FOR_CRITIC", "5"))
CV_TEXT_MAX_CHARS = int(os.getenv("CV_TEXT_MAX_CHARS", "2000"))
JOB_DESC_MAX_CHARS = int(os.getenv("JOB_DESC_MAX_CHARS", "1000"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store/")
JOBS_DATA_PATH = os.getenv("JOBS_DATA_PATH", "data/processed/jobs_dataset.json")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs/")


def get_llm(**overrides):
    """Create an OllamaLLM instance with default settings. Override any param via kwargs."""
    from langchain_ollama import OllamaLLM
    defaults = {"model": OLLAMA_MODEL, "num_predict": OLLAMA_NUM_PREDICT}
    return OllamaLLM(**(defaults | overrides))
