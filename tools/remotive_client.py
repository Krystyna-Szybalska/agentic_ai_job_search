import logging
import re

import requests


_REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"

logger = logging.getLogger(__name__)


def fetch_remotive_jobs(search: str, limit: int = 20) -> list[dict]:
    """Fetch jobs from Remotive API and return normalized results."""
    params = {"search": search, "limit": limit}
    logger.debug("GET %s params=%s", _REMOTIVE_API_URL, params)
    response = requests.get(_REMOTIVE_API_URL, params=params)
    response.raise_for_status()
    raw_jobs = response.json().get("jobs", [])
    return [normalize_remotive_job(job) for job in raw_jobs]


def normalize_remotive_job(job: dict) -> dict:
    """Map Remotive API fields to internal job format."""
    return {
        "id": str(job.get("id", "")),
        "positionName": job.get("title", ""),
        "company": job.get("company_name", ""),
        "location": job.get("candidate_required_location", ""),
        "salary": job.get("salary", "") or "N/A",
        "description": _strip_html(job.get("description", "")),
        "url": job.get("url", ""),
    }


def _strip_html(html: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", " ", html).strip()
