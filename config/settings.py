import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
TOP_K = int(os.getenv("TOP_K", "3"))
TOP_N_FOR_CRITIC = int(os.getenv("TOP_N_FOR_CRITIC", "1"))
CV_TEXT_MAX_CHARS = int(os.getenv("CV_TEXT_MAX_CHARS", "2000"))
JOB_DESC_MAX_CHARS = int(os.getenv("JOB_DESC_MAX_CHARS", "1000"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store/")
JOBS_DATA_PATH = os.getenv("JOBS_DATA_PATH", "data/processed/jobs_dataset.json")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs/")

JOB_SOURCE = os.getenv("JOB_SOURCE", "local")
REMOTIVE_SEARCH = os.getenv("REMOTIVE_SEARCH", "")
REMOTIVE_LIMIT = int(os.getenv("REMOTIVE_LIMIT", "20"))

# Model presets: thinking models need higher num_predict because internal
# reasoning tokens count toward the limit.
MODEL_PRESETS = {
    "phi3:mini":    {"num_predict": 1024, "thinking": False},
    "phi4-mini":    {"num_predict": 1024, "thinking": False},
    "qwen3:4b":     {"num_predict": 4096, "thinking": True},
    "qwen3:8b":     {"num_predict": 4096, "thinking": True},
    "gemma3:4b":    {"num_predict": 1024, "thinking": False},
    "llama3.1:8b":  {"num_predict": 1024, "thinking": False},
}

_DEFAULT_PRESET = {"num_predict": 2048, "thinking": False}


def get_llm(**overrides):
    """Create an Ollama LLM instance with settings appropriate for the model.

    Uses ChatOllama (chat API) for thinking models and OllamaLLM (generate API)
    for standard models. Returns a consistent interface: .invoke(str) -> str.
    """
    model = overrides.pop("model", OLLAMA_MODEL)
    preset = MODEL_PRESETS.get(model, _DEFAULT_PRESET)
    num_predict = overrides.pop("num_predict", preset["num_predict"])

    if preset["thinking"]:
        from langchain_ollama import ChatOllama
        chat = ChatOllama(model=model, num_predict=num_predict, **overrides)
        return _ChatLLMWrapper(chat)
    else:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=model, num_predict=num_predict, **overrides)


class _ChatLLMWrapper:
    """Wraps ChatOllama so .invoke(str) returns a plain string like OllamaLLM."""

    def __init__(self, chat_model):
        self._chat = chat_model

    def invoke(self, prompt: str) -> str:
        return self._chat.invoke(prompt).content
