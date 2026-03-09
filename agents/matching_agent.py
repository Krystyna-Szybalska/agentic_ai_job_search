import json
import re
from config.prompts import MATCHING_PROMPT


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    # Try closed fences first, then handle unclosed (truncated response)
    stripped = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL).strip()
    if stripped == text.strip():
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped, flags=re.DOTALL).strip()
    return stripped


def analyze_job(cv_text: str, job: dict, llm) -> dict:
    """Analyze a single job against the CV. Returns job dict enriched with LLM analysis."""
    prompt = MATCHING_PROMPT.format(
        cv_text=cv_text[:2000],  # truncate to avoid context overflow
        job_title=job.get("positionName", ""),
        job_company=job.get("company", ""),
        job_location=job.get("location", ""),
        job_salary=job.get("salary", "N/A"),
        job_description=job.get("description", "")[:1000],  # truncate long descriptions
    )

    raw_response = llm.invoke(prompt)
    result = dict(job)

    try:
        parsed = json.loads(_strip_code_fences(raw_response))
        result["llm_score"] = _clamp_score(parsed.get("score", 0))
        result["matching_skills"] = _sanitize_string_list(parsed.get("matching_skills", []))
        result["missing_skills"] = _sanitize_string_list(parsed.get("missing_skills", []))
        result["summary"] = str(parsed.get("summary", ""))[:500]
    except (json.JSONDecodeError, ValueError):
        result.update(_parse_partial_json(raw_response))

    return result


def _clamp_score(value) -> int:
    """Ensure score is an integer in range 1-10."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, score))


def _sanitize_string_list(items) -> list[str]:
    """Ensure the value is a list of short strings."""
    if not isinstance(items, list):
        return []
    return [str(item)[:100] for item in items if isinstance(item, (str, int, float))]


def _parse_partial_json(text: str) -> dict:
    """Best-effort extraction from truncated/malformed LLM JSON output."""
    cleaned = _strip_code_fences(text)
    result: dict = {}

    score_m = re.search(r'"score"\s*:\s*(\d+)', cleaned)
    result["llm_score"] = int(score_m.group(1)) if score_m else 0

    result["matching_skills"] = _extract_string_list(cleaned, "matching_skills")
    result["missing_skills"] = _extract_string_list(cleaned, "missing_skills")

    summary_m = re.search(r'"summary"\s*:\s*"([^"]*)', cleaned)
    result["summary"] = summary_m.group(1) if summary_m else ""

    return result


def _extract_string_list(text: str, key: str) -> list[str]:
    """Extract complete quoted strings from a possibly-truncated JSON array."""
    m = re.search(rf'"{key}"\s*:\s*\[', text)
    if not m:
        return []
    after_bracket = text[m.end():]
    return re.findall(r'"([^"]+)"', after_bracket.split("]")[0])
