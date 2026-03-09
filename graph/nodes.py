from graph.state import JobSearchState
from tools.pdf_parser import extract_text_from_pdf
from embeddings.embedder import get_embedder, embed_text
from embeddings.vector_store import VectorStore
from agents.matching_agent import analyze_job
from agents.critic_agent import critique_matches
from utils.report_generator import generate_report
from config.settings import (
    OLLAMA_MODEL, TOP_K, TOP_N_FOR_CRITIC, VECTOR_STORE_PATH, OUTPUTS_DIR
)
from langchain_ollama import OllamaLLM


def parse_cv(state: JobSearchState) -> dict:
    cv_text = extract_text_from_pdf(state["cv_path"])
    return {"cv_text": cv_text}


def embed_cv(state: JobSearchState) -> dict:
    embedder = get_embedder()
    embedding = embed_text(state["cv_text"], embedder)
    return {"cv_embedding": embedding}


def retrieve_jobs(state: JobSearchState) -> dict:
    store = VectorStore(dimension=384)
    store.load(VECTOR_STORE_PATH)
    results = store.query(state["cv_embedding"], k=TOP_K)
    return {"retrieved_jobs": results}


def matching_agent(state: JobSearchState) -> dict:
    llm = OllamaLLM(model=OLLAMA_MODEL, num_predict=1024)
    matched = []
    for job in state["retrieved_jobs"]:
        print(f"  Analyzing: {job.get('positionName', '')} at {job.get('company', '')}...")
        result = analyze_job(state["cv_text"], job, llm)
        matched.append(result)
    matched.sort(key=lambda j: j.get("llm_score", 0), reverse=True)
    return {"matched_jobs": matched}


def critic_agent(state: JobSearchState) -> dict:
    llm = OllamaLLM(model=OLLAMA_MODEL, num_predict=1024)
    top_jobs = state["matched_jobs"][:TOP_N_FOR_CRITIC]
    validated, feedback = critique_matches(state["cv_text"], top_jobs, llm)
    return {"validated_jobs": validated, "critic_feedback": feedback}


def human_review(state: JobSearchState) -> dict:
    # This node is reached after the INTERRUPT in graph_builder.
    # The graph resumes here after user input is provided via main.py.
    return {}


def generate_report_node(state: JobSearchState) -> dict:
    path = generate_report(
        state["validated_jobs"],
        critic_feedback=state.get("critic_feedback", ""),
        cv_path=state.get("cv_path", ""),
        output_dir=OUTPUTS_DIR,
    )
    print(f"\nReport saved to: {path}")
    return {"report_path": path}
