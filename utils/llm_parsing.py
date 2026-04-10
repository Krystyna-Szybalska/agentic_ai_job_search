"""Shared utilities for parsing LLM JSON output."""

import json
import re


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    stripped = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL).strip()
    if stripped == text.strip():
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped, flags=re.DOTALL).strip()
    return stripped


def parse_llm_json(raw_response: str) -> dict | None:
    """Parse JSON from LLM response, stripping code fences. Returns None on failure."""
    try:
        return json.loads(strip_code_fences(raw_response))
    except (json.JSONDecodeError, ValueError):
        return None


def extract_json_field(text: str, key: str, fallback=""):
    """Extract a single JSON field value via regex from malformed output."""
    cleaned = strip_code_fences(text)
    m = re.search(rf'"{key}"\s*:\s*"([^"]*)', cleaned)
    return m.group(1) if m else fallback


def extract_json_number(text: str, key: str, fallback: int = 0) -> int:
    """Extract a single numeric JSON field via regex from malformed output."""
    cleaned = strip_code_fences(text)
    m = re.search(rf'"{key}"\s*:\s*(\d+)', cleaned)
    return int(m.group(1)) if m else fallback


def extract_string_list(text: str, key: str) -> list[str]:
    """Extract complete quoted strings from a possibly-truncated JSON array."""
    cleaned = strip_code_fences(text)
    m = re.search(rf'"{key}"\s*:\s*\[', cleaned)
    if not m:
        return []
    after_bracket = cleaned[m.end():]
    return re.findall(r'"([^"]+)"', after_bracket.split("]")[0])


def clamp_score(value, low: int = 0, high: int = 10) -> int:
    """Ensure score is an integer in the given range."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(low, min(high, score))


def sanitize_string_list(items) -> list[str]:
    """Ensure the value is a list of short strings."""
    if not isinstance(items, list):
        return []
    return [str(item)[:100] for item in items if isinstance(item, (str, int, float))]
