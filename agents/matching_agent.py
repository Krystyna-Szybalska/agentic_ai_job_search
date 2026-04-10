from config.prompts import MATCHING_PROMPT
from config.settings import CV_TEXT_MAX_CHARS, JOB_DESC_MAX_CHARS
from utils.llm_parsing import (
    parse_llm_json,
    extract_json_number,
    extract_json_field,
    extract_string_list,
    clamp_score,
    sanitize_string_list,
)


def analyze_job(cv_text: str, job: dict, llm) -> dict:
    """Analyze a single job against the CV. Returns job dict enriched with LLM analysis."""
    prompt = MATCHING_PROMPT.format(
        cv_text=cv_text[:CV_TEXT_MAX_CHARS],
        job_title=job.get("positionName", ""),
        job_company=job.get("company", ""),
        job_location=job.get("location", ""),
        job_salary=job.get("salary", "N/A"),
        job_description=job.get("description", "")[:JOB_DESC_MAX_CHARS],
    )

    raw_response = llm.invoke(prompt)
    result = dict(job)

    parsed = parse_llm_json(raw_response)
    if parsed is not None:
        result["llm_score"] = clamp_score(parsed.get("score", 0))
        result["matching_skills"] = sanitize_string_list(parsed.get("matching_skills", []))
        result["missing_skills"] = sanitize_string_list(parsed.get("missing_skills", []))
        result["summary"] = str(parsed.get("summary", ""))[:500]
    else:
        result["llm_score"] = extract_json_number(raw_response, "score")
        result["matching_skills"] = extract_string_list(raw_response, "matching_skills")
        result["missing_skills"] = extract_string_list(raw_response, "missing_skills")
        result["summary"] = extract_json_field(raw_response, "summary")

    return result
