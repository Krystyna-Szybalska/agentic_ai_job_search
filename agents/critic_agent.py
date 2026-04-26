import logging

from config.prompts import CRITIC_PROMPT
from config.settings import CV_TEXT_MAX_CHARS
from utils.llm_parsing import parse_llm_json, extract_json_field, strip_code_fences

logger = logging.getLogger(__name__)


def critique_matches(cv_text: str, matched_jobs: list[dict], llm) -> tuple[list[dict], str]:
    """Validate and rerank matched jobs. Returns (validated_jobs, feedback)."""
    summary_lines = [
        f"- [{job['id']}] {job.get('positionName', '')} at {job.get('company', '')} — score: {job.get('llm_score', 0)}"
        for job in matched_jobs
    ]
    matched_jobs_summary = "\n".join(summary_lines)

    prompt = CRITIC_PROMPT.format(
        cv_text=cv_text[:CV_TEXT_MAX_CHARS],
        matched_jobs_summary=matched_jobs_summary,
    )

    logger.debug("Prompt to LLM (critic):\n%s", prompt)
    raw_response = llm.invoke(prompt)
    logger.debug("Raw LLM response (critic):\n%s", raw_response)
    jobs_by_id = {job["id"]: job for job in matched_jobs}

    parsed = parse_llm_json(raw_response)
    if parsed is not None:
        feedback = str(parsed.get("feedback", ""))[:1000]
        ranking = parsed.get("suggested_ranking", [])
        if not isinstance(ranking, list):
            ranking = []
        reordered = []
        for job_id in ranking:
            if str(job_id) in jobs_by_id:
                reordered.append(jobs_by_id[str(job_id)])
        mentioned = {str(r) for r in ranking}
        for job in matched_jobs:
            if job["id"] not in mentioned:
                reordered.append(job)
        return reordered, feedback

    feedback = extract_json_field(raw_response, "feedback")
    if not feedback:
        feedback = strip_code_fences(raw_response)[:200]
    return matched_jobs, feedback
