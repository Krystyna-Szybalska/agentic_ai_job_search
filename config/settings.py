import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "10"))
TOP_N_FOR_CRITIC = int(os.getenv("TOP_N_FOR_CRITIC", "5"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store/")
JOBS_DATA_PATH = os.getenv("JOBS_DATA_PATH", "data/processed/jobs_dataset.json")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs/")
