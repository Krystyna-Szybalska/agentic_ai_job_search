# Remotive API Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Remotive API as an alternative job source, controlled by a `JOB_SOURCE` config parameter, while preserving the existing local FAISS pipeline.

**Architecture:** A new `tools/remotive_client.py` module fetches and normalizes jobs from the Remotive API. The graph conditionally routes `parse_cv` to either the local embed+vector path or a new `fetch_remote_jobs` node. Everything from `matching_agent` onward is unchanged.

**Tech Stack:** Python `requests` (new dependency), `re` for HTML stripping, LangGraph conditional edges.

**Design doc:** `docs/plans/2026-04-11-remotive-api-design.md`

---

### Task 1: Add config settings

**Files:**
- Modify: `config/settings.py:1-15`
- Modify: `.env.example:1-19`
- Modify: `.env` (if present)

**Step 1: Add new settings to `config/settings.py`**

After line 15 (`OUTPUTS_DIR = ...`), add:

```python
JOB_SOURCE = os.getenv("JOB_SOURCE", "local")
REMOTIVE_SEARCH = os.getenv("REMOTIVE_SEARCH", "")
REMOTIVE_LIMIT = int(os.getenv("REMOTIVE_LIMIT", "20"))
```

**Step 2: Add new settings to `.env.example`**

Add a new section after the `HF_HUB_OFFLINE` line:

```env
# Job source: "local" (FAISS vector store) or "remotive" (Remotive API)
JOB_SOURCE=local

# Remotive API settings (only used when JOB_SOURCE=remotive)
REMOTIVE_SEARCH=python developer
REMOTIVE_LIMIT=20
```

**Step 3: Commit**

```bash
git add config/settings.py .env.example
git commit -m "feat: add JOB_SOURCE, REMOTIVE_SEARCH, REMOTIVE_LIMIT config settings"
```

---

### Task 2: Create Remotive client — tests first

**Files:**
- Create: `tests/test_remotive_client.py`

**Step 1: Write failing tests for `fetch_remotive_jobs` and `normalize_remotive_job`**

```python
import pytest
from unittest.mock import patch, Mock


SAMPLE_API_RESPONSE = {
    "jobs": [
        {
            "id": 12345,
            "url": "https://remotive.com/remote-jobs/software-dev/senior-python-12345",
            "title": "Senior Python Developer",
            "company_name": "Acme Corp",
            "category": "Software Development",
            "tags": ["python", "django"],
            "job_type": "full_time",
            "publication_date": "2026-04-01T00:00:00",
            "candidate_required_location": "Worldwide",
            "salary": "$100k - $150k",
            "description": "<p>We are looking for a <strong>Python developer</strong>.</p><ul><li>Django</li><li>REST APIs</li></ul>",
        },
        {
            "id": 67890,
            "url": "https://remotive.com/remote-jobs/software-dev/ml-engineer-67890",
            "title": "ML Engineer",
            "company_name": "DataCo",
            "category": "Software Development",
            "tags": ["python", "ml"],
            "job_type": "full_time",
            "publication_date": "2026-04-02T00:00:00",
            "candidate_required_location": "Europe",
            "salary": "",
            "description": "<div>Build ML pipelines.</div>",
        },
    ]
}


class TestNormalizeRemotiveJob:
    def test_maps_fields_to_internal_format(self):
        from tools.remotive_client import normalize_remotive_job

        raw = SAMPLE_API_RESPONSE["jobs"][0]
        result = normalize_remotive_job(raw)

        assert result["positionName"] == "Senior Python Developer"
        assert result["company"] == "Acme Corp"
        assert result["location"] == "Worldwide"
        assert result["salary"] == "$100k - $150k"
        assert result["id"] == "12345"
        assert result["url"] == raw["url"]

    def test_strips_html_from_description(self):
        from tools.remotive_client import normalize_remotive_job

        raw = SAMPLE_API_RESPONSE["jobs"][0]
        result = normalize_remotive_job(raw)

        assert "<p>" not in result["description"]
        assert "<strong>" not in result["description"]
        assert "<ul>" not in result["description"]
        assert "Python developer" in result["description"]

    def test_handles_empty_salary(self):
        from tools.remotive_client import normalize_remotive_job

        raw = SAMPLE_API_RESPONSE["jobs"][1]
        result = normalize_remotive_job(raw)

        assert result["salary"] == "N/A"


class TestFetchRemotiveJobs:
    @patch("tools.remotive_client.requests.get")
    def test_returns_normalized_jobs(self, mock_get):
        from tools.remotive_client import fetch_remotive_jobs

        mock_response = Mock()
        mock_response.json.return_value = SAMPLE_API_RESPONSE
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        results = fetch_remotive_jobs("python developer", limit=10)

        assert len(results) == 2
        assert results[0]["positionName"] == "Senior Python Developer"
        assert results[1]["positionName"] == "ML Engineer"
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "search=python+developer" in call_url or "search=python%20developer" in call_url

    @patch("tools.remotive_client.requests.get")
    def test_raises_on_http_error(self, mock_get):
        from tools.remotive_client import fetch_remotive_jobs

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("503 Service Unavailable")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="503"):
            fetch_remotive_jobs("python", limit=5)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_remotive_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.remotive_client'`

**Step 3: Commit**

```bash
git add tests/test_remotive_client.py
git commit -m "test: add failing tests for remotive client"
```

---

### Task 3: Implement Remotive client

**Files:**
- Create: `tools/remotive_client.py`

**Step 1: Implement the module**

```python
import re
import requests


_REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"


def fetch_remotive_jobs(search: str, limit: int = 20) -> list[dict]:
    """Fetch jobs from Remotive API and return normalized results."""
    response = requests.get(
        _REMOTIVE_API_URL,
        params={"search": search, "limit": limit},
    )
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
```

**Step 2: Run the tests**

Run: `pytest tests/test_remotive_client.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add tools/remotive_client.py
git commit -m "feat: add Remotive API client with field normalization"
```

---

### Task 4: Add `fetch_remote_jobs` node — test first

**Files:**
- Modify: `tests/test_nodes.py`

**Step 1: Add a test for the new node**

Append to `tests/test_nodes.py`:

```python
def test_fetch_remote_jobs_node():
    from graph.nodes import fetch_remote_jobs
    state = _make_state(cv_text="Python ML engineer")
    fake_jobs = [
        {"id": "1", "positionName": "ML Engineer", "company": "Co", "location": "Remote",
         "salary": "N/A", "description": "Build ML stuff", "url": "https://example.com/1"},
    ]
    with patch("graph.nodes.fetch_remotive_jobs", return_value=fake_jobs):
        result = fetch_remote_jobs(state)
    assert len(result["retrieved_jobs"]) == 1
    assert result["retrieved_jobs"][0]["positionName"] == "ML Engineer"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_nodes.py::test_fetch_remote_jobs_node -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_remote_jobs'`

**Step 3: Commit**

```bash
git add tests/test_nodes.py
git commit -m "test: add failing test for fetch_remote_jobs node"
```

---

### Task 5: Implement `fetch_remote_jobs` node

**Files:**
- Modify: `graph/nodes.py:1-11` (imports) and append new function

**Step 1: Add imports to `graph/nodes.py`**

Add to the imports section:

```python
from tools.remotive_client import fetch_remotive_jobs
from config.settings import REMOTIVE_SEARCH, REMOTIVE_LIMIT
```

Note: `fetch_remotive_jobs` is imported with its full name to match the mock path in the test.

**Step 2: Add the node function**

Append to `graph/nodes.py`:

```python
def fetch_remote_jobs(state: JobSearchState) -> dict:
    print(f"  Fetching jobs from Remotive API (search: '{REMOTIVE_SEARCH}')...")
    jobs = fetch_remotive_jobs(REMOTIVE_SEARCH, limit=REMOTIVE_LIMIT)
    print(f"  Found {len(jobs)} remote jobs.")
    return {"retrieved_jobs": jobs}
```

**Step 3: Run all node tests**

Run: `pytest tests/test_nodes.py -v`
Expected: All tests PASS (including new `test_fetch_remote_jobs_node`)

**Step 4: Commit**

```bash
git add graph/nodes.py
git commit -m "feat: add fetch_remote_jobs node"
```

---

### Task 6: Add conditional graph routing — test first

**Files:**
- Create: `tests/test_graph_builder.py`

**Step 1: Write failing tests for graph routing**

```python
from unittest.mock import patch


class TestGraphRouting:
    @patch("graph.graph_builder.JOB_SOURCE", "local")
    def test_local_source_includes_embed_and_retrieve(self):
        from graph.graph_builder import build_graph
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "embed_cv" in node_names
        assert "retrieve_jobs" in node_names

    @patch("graph.graph_builder.JOB_SOURCE", "remotive")
    def test_remotive_source_includes_fetch_remote(self):
        from graph.graph_builder import build_graph
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "fetch_remote_jobs" in node_names
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_builder.py -v`
Expected: FAIL — `ImportError: cannot import name 'JOB_SOURCE'`

**Step 3: Commit**

```bash
git add tests/test_graph_builder.py
git commit -m "test: add failing tests for conditional graph routing"
```

---

### Task 7: Implement conditional graph routing

**Files:**
- Modify: `graph/graph_builder.py`

**Step 1: Rewrite `graph_builder.py` with conditional routing**

Replace the full contents of `graph/graph_builder.py` with:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import JobSearchState
from graph.nodes import (
    parse_cv,
    embed_cv,
    retrieve_jobs,
    matching_agent,
    critic_agent,
    generate_report_node,
    fetch_remote_jobs,
)
from config.settings import JOB_SOURCE


def build_graph():
    builder = StateGraph(JobSearchState)

    builder.add_node("parse_cv", parse_cv)
    builder.add_node("matching_agent", matching_agent)
    builder.add_node("critic_agent", critic_agent)
    builder.add_node("generate_report", generate_report_node)

    if JOB_SOURCE == "remotive":
        builder.add_node("fetch_remote_jobs", fetch_remote_jobs)
        builder.add_edge(START, "parse_cv")
        builder.add_edge("parse_cv", "fetch_remote_jobs")
        builder.add_edge("fetch_remote_jobs", "matching_agent")
    else:
        builder.add_node("embed_cv", embed_cv)
        builder.add_node("retrieve_jobs", retrieve_jobs)
        builder.add_edge(START, "parse_cv")
        builder.add_edge("parse_cv", "embed_cv")
        builder.add_edge("embed_cv", "retrieve_jobs")
        builder.add_edge("retrieve_jobs", "matching_agent")

    builder.add_edge("matching_agent", "critic_agent")
    builder.add_edge("critic_agent", "generate_report")
    builder.add_edge("generate_report", END)

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["generate_report"],
    )
    return graph
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add graph/graph_builder.py
git commit -m "feat: add conditional graph routing for local vs remotive job source"
```

---

### Task 8: Update `.env.example`, `.env`, and CLAUDE.md

**Files:**
- Modify: `.env.example`
- Modify: `.env`
- Modify: `CLAUDE.md`

**Step 1: Add Remotive settings to `.env.example`**

Append after the `HF_HUB_OFFLINE` line:

```env

# Job source: "local" (FAISS vector store) or "remotive" (Remotive API)
JOB_SOURCE=local

# Remotive API settings (only used when JOB_SOURCE=remotive)
REMOTIVE_SEARCH=python developer
REMOTIVE_LIMIT=20
```

**Step 2: Add Remotive settings to `.env`**

Append:

```env
JOB_SOURCE=local
REMOTIVE_SEARCH=
REMOTIVE_LIMIT=20
```

**Step 3: Update `CLAUDE.md`**

Add `tools/remotive_client.py` to the architecture table with description: `Remotive API client (fetch + normalize jobs)`.

Update the pipeline description to mention the branching: `parse_cv -> (embed_cv -> retrieve_jobs | fetch_remote_jobs) -> matching_agent -> ...`

Add `requests` to the tech stack.

**Step 4: Commit**

```bash
git add .env.example .env CLAUDE.md
git commit -m "docs: update config and docs for Remotive API integration"
```

---

### Task 9: Add `requests` dependency

**Files:**
- Check: `requirements.txt` or `pyproject.toml` (whichever exists)

**Step 1: Determine dependency file**

Run: `ls requirements.txt pyproject.toml 2>/dev/null`

**Step 2: Add `requests` to the dependency file**

Add `requests` to the appropriate file.

**Step 3: Commit**

```bash
git add requirements.txt  # or pyproject.toml
git commit -m "feat: add requests dependency for Remotive API"
```

---

### Task 10: Run full test suite and verify

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify no import errors in remotive mode**

Run: `python -c "from config.settings import JOB_SOURCE; print(f'JOB_SOURCE={JOB_SOURCE}')"`
Expected: prints `JOB_SOURCE=local`

Run: `python -c "from tools.remotive_client import fetch_remotive_jobs; print('import OK')"`
Expected: prints `import OK`
