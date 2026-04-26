import logging

from graph.state import JobSearchState
from tools.pdf_parser import extract_text_from_pdf
from embeddings.embedder import get_embedder, embed_text
from embeddings.vector_store import VectorStore
from agents.matching_agent import analyze_job
from agents.critic_agent import critique_matches
from utils.report_generator import generate_report
from utils.decorators import log_step
from tools.remotive_client import fetch_remotive_jobs
from config.settings import (
    TOP_K, TOP_N_FOR_CRITIC, VECTOR_STORE_PATH, OUTPUTS_DIR,
    EMBEDDING_DIMENSION, REMOTIVE_LIMIT, get_llm,
)

logger = logging.getLogger(__name__)


@log_step("Parse CV")
def parse_cv(state: JobSearchState) -> dict:
    cv_text = extract_text_from_pdf(state["cv_path"])
    return {"cv_text": cv_text}


@log_step("Embed CV")
def embed_cv(state: JobSearchState) -> dict:
    embedder = get_embedder()
    embedding = embed_text(state["cv_text"], embedder)
    return {"cv_embedding": embedding}


@log_step("Retrieve jobs (vector search)")
def retrieve_jobs(state: JobSearchState) -> dict:
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    store.load(VECTOR_STORE_PATH)
    results = store.query(state["cv_embedding"], k=TOP_K)
    logger.info("Retrieved %d jobs from vector store", len(results))
    return {"retrieved_jobs": results}


@log_step("Matching agent")
def matching_agent(state: JobSearchState) -> dict:
    llm = get_llm()
    matched = []
    total = len(state["retrieved_jobs"])
    for i, job in enumerate(state["retrieved_jobs"], start=1):
        logger.debug("Analyzing %d/%d: %s at %s", i, total,
                     job.get("positionName", ""), job.get("company", ""))
        result = analyze_job(state["cv_text"], job, llm)
        matched.append(result)
    matched.sort(key=lambda j: j.get("llm_score", 0), reverse=True)
    if matched:
        top = matched[0]
        logger.info("Top match: %s (%s/10)", top.get("positionName", ""), top.get("llm_score", "?"))
    return {"matched_jobs": matched}


@log_step("Critic agent")
def critic_agent(state: JobSearchState) -> dict:
    llm = get_llm()
    top_jobs = state["matched_jobs"][:TOP_N_FOR_CRITIC]
    logger.info("Critic reviewing top %d matches", len(top_jobs))
    validated, feedback = critique_matches(state["cv_text"], top_jobs, llm)
    return {"validated_jobs": validated, "critic_feedback": feedback}


@log_step("Generate report")
def generate_report_node(state: JobSearchState) -> dict:
    path = generate_report(
        state["validated_jobs"],
        critic_feedback=state.get("critic_feedback", ""),
        cv_path=state.get("cv_path", ""),
        output_dir=OUTPUTS_DIR,
    )
    logger.info("Report saved to: %s", path)
    return {"report_path": path}


@log_step("Fetch jobs from Remotive")
def fetch_remote_jobs(state: JobSearchState) -> dict:
    search = state["remotive_search"]
    jobs = fetch_remotive_jobs(search, limit=REMOTIVE_LIMIT)
    logger.info("Found %d remote jobs for query '%s'", len(jobs), search)
    return {"retrieved_jobs": jobs}
