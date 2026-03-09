import json
import re
from config.prompts import CRITIC_PROMPT


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    # Try closed fences first, then handle unclosed (truncated response)
    stripped = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL).strip()
    if stripped == text.strip():
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped, flags=re.DOTALL).strip()
    return stripped


def critique_matches(cv_text: str, matched_jobs: list[dict], llm) -> tuple[list[dict], str]:
    """Validate and rerank matched jobs. Returns (validated_jobs, feedback)."""
    summary_lines = [
        f"- [{job['id']}] {job.get('positionName', '')} at {job.get('company', '')} — score: {job.get('llm_score', 0)}"
        for job in matched_jobs
    ]
    matched_jobs_summary = "\n".join(summary_lines)

    prompt = CRITIC_PROMPT.format(
        cv_text=cv_text[:1000],
        matched_jobs_summary=matched_jobs_summary,
    )

    raw_response = llm.invoke(prompt)
    jobs_by_id = {job["id"]: job for job in matched_jobs}

    try:
        parsed = json.loads(_strip_code_fences(raw_response))
        feedback = str(parsed.get("feedback", ""))[:1000]
        ranking = parsed.get("suggested_ranking", [])
        if not isinstance(ranking, list):
            ranking = []
        # Reorder jobs according to critic's suggested ranking
        reordered = []
        for job_id in ranking:
            if str(job_id) in jobs_by_id:
                reordered.append(jobs_by_id[str(job_id)])
        # Append any jobs not mentioned in ranking
        mentioned = {str(r) for r in ranking}
        for job in matched_jobs:
            if job["id"] not in mentioned:
                reordered.append(job)
        return reordered, feedback
    except (json.JSONDecodeError, KeyError):
        cleaned = _strip_code_fences(raw_response)
        feedback_m = re.search(r'"feedback"\s*:\s*"([^"]*)', cleaned)
        feedback = feedback_m.group(1) if feedback_m else cleaned[:200]
        return matched_jobs, feedback
