# Verbose Logging, Remotive HITL & Report URL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> **Note:** The user handles all `git commit` operations manually. Where this plan mentions "Commit point", treat it as a logical pause to inform the user that a chunk of related work is ready to commit — do NOT run `git commit`.

**Goal:** Add verbose logging (toggle with `-v`), human-in-the-loop confirmation before Remotive API calls (with editable search query), and clickable job offer links in the markdown report.

**Architecture:** Stdlib `logging` with a custom `@log_step` decorator on graph nodes. New `remotive_search` field in `JobSearchState` enables runtime mutability of the API query. `interrupt_before` is extended (conditionally on `JOB_SOURCE`) to pause before `fetch_remote_jobs`; the main loop dispatches on `state.next` to handle either pause point. Report generator reads `url` (with `externalApplyLink` fallback) and emits a markdown link.

**Tech Stack:** Python stdlib (`logging`, `argparse`, `functools`, `time`), pytest with `caplog` and `monkeypatch`, existing LangGraph `interrupt_before` mechanism.

**Reference design doc:** [docs/plans/2026-04-26-verbose-logging-and-remotive-hitl-design.md](2026-04-26-verbose-logging-and-remotive-hitl-design.md)

---

## Task 1: Create logging configuration

**Files:**
- Create: `utils/logging_config.py`
- Test: `tests/test_logging_config.py`

**Step 1: Write the failing test**

```python
# tests/test_logging_config.py
import logging
from utils.logging_config import setup_logging


def test_setup_logging_default_is_info():
    setup_logging(verbose=False)
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_verbose_is_debug():
    setup_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_silences_urllib3():
    setup_logging(verbose=True)
    assert logging.getLogger("urllib3").level == logging.WARNING
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_logging_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.logging_config'`

**Step 3: Implement `utils/logging_config.py`**

```python
"""Central logging configuration for the job search pipeline."""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger. Called once from main.py at startup.

    verbose=True sets DEBUG level (LLM prompts, raw responses, HTTP details).
    verbose=False sets INFO level (node entry/exit, key counts).
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_logging_config.py -v`
Expected: 3 passed

**Step 5: Commit point** — `utils/logging_config.py`, `tests/test_logging_config.py` ready to commit.

---

## Task 2: Create `@log_step` decorator

**Files:**
- Create: `utils/decorators.py`
- Test: `tests/test_decorators.py`

**Step 1: Write the failing tests**

```python
# tests/test_decorators.py
import logging
import pytest
from utils.decorators import log_step


def test_log_step_logs_start_and_finish(caplog):
    @log_step("My step")
    def my_func():
        return "result"

    with caplog.at_level(logging.INFO):
        result = my_func()

    assert result == "result"
    messages = [r.message for r in caplog.records]
    assert any("My step started" in m for m in messages)
    assert any("My step finished" in m for m in messages)


def test_log_step_propagates_return_value():
    @log_step("Echo")
    def echo(x):
        return x * 2

    assert echo(21) == 42


def test_log_step_logs_exception_and_reraises(caplog):
    @log_step("Boom")
    def bad():
        raise ValueError("kaboom")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError, match="kaboom"):
            bad()

    assert any("Boom failed" in r.message for r in caplog.records)


def test_log_step_uses_function_name_when_no_name_given(caplog):
    @log_step()
    def my_named_func():
        return None

    with caplog.at_level(logging.INFO):
        my_named_func()

    assert any("my_named_func started" in r.message for r in caplog.records)


def test_log_step_logger_uses_caller_module(caplog):
    @log_step("Test")
    def f():
        return None

    with caplog.at_level(logging.INFO):
        f()

    # logger name should be the test module's name, not 'utils.decorators'
    assert any(r.name == __name__ for r in caplog.records)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_decorators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.decorators'`

**Step 3: Implement `utils/decorators.py`**

```python
"""Reusable function decorators."""

import functools
import logging
import time


def log_step(name: str | None = None):
    """Decorator that logs entry/exit and timing of a function.

    The logger is resolved from the wrapped function's module so messages
    are attributed to the calling module (e.g. 'graph.nodes'), not 'utils.decorators'.
    Exceptions are logged with stack trace via logger.exception() and re-raised.
    """
    def decorator(func):
        step_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.info("→ %s started", step_name)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("✓ %s finished (%.2fs)", step_name, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                logger.exception("✗ %s failed (%.2fs)", step_name, elapsed)
                raise
        return wrapper
    return decorator
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_decorators.py -v`
Expected: 5 passed

**Step 5: Commit point** — `utils/decorators.py`, `tests/test_decorators.py` ready to commit.

---

## Task 3: Replace `sys.argv` parsing in `main.py` with argparse + wire up `setup_logging`

**Files:**
- Modify: `main.py` (lines 64-77 — the `if __name__ == "__main__":` block)

**Step 1: Read current `main.py`**

Confirm the existing flow at the bottom:
```python
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/cv.pdf")
        sys.exit(1)
    cv_file = sys.argv[1]
    ...
    main(cv_file)
```

**Step 2: Modify `main.py`**

Replace the bottom block AND extend the `main()` signature to accept `search_override` (used in Task 11 for `remotive_search` state init). The `search_override` is plumbed through but unused for now — prepares for Task 11 without breaking flow.

Add at top of file (after existing imports):
```python
import argparse
from utils.logging_config import setup_logging
```

Replace `def main(cv_path: str) -> None:` signature with:
```python
def main(cv_path: str, search_override: str | None = None) -> None:
```

(Body stays unchanged for now — `search_override` will be used in Task 11.)

Replace the `if __name__ == "__main__":` block at the bottom with:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match a CV against job listings using local LLM + vector search.",
    )
    parser.add_argument("cv_path", help="Path to the CV file (PDF).")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging, including LLM prompts and raw responses.",
    )
    parser.add_argument(
        "-s", "--search",
        default=None,
        help="Override REMOTIVE_SEARCH from .env (only used when JOB_SOURCE=remotive).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.cv_path):
        sys.exit(f"Error: file not found: {args.cv_path}")
    if not args.cv_path.lower().endswith(".pdf"):
        sys.exit("Error: file must be a PDF (.pdf)")

    setup_logging(verbose=args.verbose)
    main(args.cv_path, search_override=args.search)
```

**Step 3: Manually verify CLI**

Run: `python main.py --help`
Expected output: argparse-generated help text showing `cv_path`, `-v/--verbose`, `-s/--search`.

Run: `python main.py nonexistent.pdf`
Expected: `Error: file not found: nonexistent.pdf`, exit code 1.

Run: `python main.py main.py`
Expected: `Error: file must be a PDF (.pdf)`.

**Step 4: Verify nothing else broke**

Run: `pytest tests/ -v`
Expected: all existing tests still pass (we didn't change `main()` body, only entry point).

**Step 5: Commit point** — `main.py` ready to commit.

---

## Task 4: Replace `print()` in `graph/nodes.py` with logger + `@log_step`

**Files:**
- Modify: `graph/nodes.py`
- Modify: `tests/test_nodes.py` (only if existing tests check for stdout — likely none)

**Context:** Current node functions have scattered `print()` calls. We replace them with `logger.info()`/`logger.debug()` and apply `@log_step` for entry/exit messages. This removes [nodes.py:37](graph/nodes.py#L37), [nodes.py:47](graph/nodes.py#L47), [nodes.py:59](graph/nodes.py#L59), [nodes.py:64-66](graph/nodes.py#L64-L66) print calls.

**Step 1: Verify no existing test asserts on stdout from nodes**

Run: `grep -n "print\|capsys\|capfd" tests/test_nodes.py`
Expected: no matches (or only matches unrelated to nodes.py prints).
If there are matches, those tests need updating — note them.

**Step 2: Modify `graph/nodes.py`**

Add imports at the top:
```python
import logging
from utils.decorators import log_step
```

Add module logger after imports:
```python
logger = logging.getLogger(__name__)
```

Apply `@log_step` to each node and replace `print()` calls:

```python
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
    jobs = fetch_remotive_jobs(REMOTIVE_SEARCH, limit=REMOTIVE_LIMIT)
    logger.info("Found %d remote jobs", len(jobs))
    return {"retrieved_jobs": jobs}
```

(Note: `fetch_remote_jobs` still uses `REMOTIVE_SEARCH` from config — Task 9 changes it to read from state.)

**Step 3: Run existing node tests**

Run: `pytest tests/test_nodes.py -v`
Expected: all pass. If `@log_step` broke anything, the most likely culprit is `functools.wraps` not propagating something LangGraph needs — debug at that point.

**Step 4: Manual smoke test (optional)**

Run: `python main.py path/to/sample.pdf -v`
Expected: see formatted INFO/DEBUG logs in terminal with timestamps and node names. No `print()` clutter from removed lines.

**Step 5: Commit point** — `graph/nodes.py` ready to commit.

---

## Task 5: Add DEBUG logs to `agents/matching_agent.py`

**Files:**
- Modify: `agents/matching_agent.py`
- Test: `tests/test_matching_agent.py` (extend with caplog assertion)

**Step 1: Add a failing test for prompt logging**

In `tests/test_matching_agent.py`, add:

```python
import logging


def test_analyze_job_logs_prompt_and_response_at_debug(caplog):
    from agents.matching_agent import analyze_job
    cv = "Python developer"
    job = {"positionName": "Dev", "company": "Acme", "location": "Remote",
           "salary": "N/A", "description": "Role", "id": "1"}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"score": 7, "matching_skills": [], "missing_skills": [], "summary": "ok"}'

    with caplog.at_level(logging.DEBUG, logger="agents.matching_agent"):
        analyze_job(cv, job, mock_llm)

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("Prompt" in m for m in debug_messages)
    assert any("Raw" in m or "response" in m.lower() for m in debug_messages)
```

(Adjust the `MagicMock` import at the top of the test file if not already present.)

**Step 2: Run test to verify failure**

Run: `pytest tests/test_matching_agent.py::test_analyze_job_logs_prompt_and_response_at_debug -v`
Expected: FAIL — no debug logs match.

**Step 3: Modify `agents/matching_agent.py`**

Add at the top:
```python
import logging

logger = logging.getLogger(__name__)
```

In `analyze_job`, wrap the LLM call with debug logs:

```python
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

    logger.debug("Prompt to LLM (matching, job=%s):\n%s", job.get("id", "?"), prompt)
    raw_response = llm.invoke(prompt)
    logger.debug("Raw LLM response (matching, job=%s):\n%s", job.get("id", "?"), raw_response)

    result = dict(job)
    # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_matching_agent.py -v`
Expected: all pass, including new test.

**Step 5: Commit point** — `agents/matching_agent.py`, `tests/test_matching_agent.py` ready to commit.

---

## Task 6: Add DEBUG logs to `agents/critic_agent.py`

**Files:**
- Modify: `agents/critic_agent.py`
- Test: `tests/test_critic_agent.py` (extend)

**Step 1: Add a failing test analogous to Task 5**

```python
def test_critique_matches_logs_prompt_and_response_at_debug(caplog):
    import logging
    from agents.critic_agent import critique_matches
    cv = "Python developer"
    matched = [{"id": "1", "positionName": "Dev", "company": "Acme", "llm_score": 7}]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"verdict": "approved", "feedback": "ok", "suggested_ranking": ["1"]}'

    with caplog.at_level(logging.DEBUG, logger="agents.critic_agent"):
        critique_matches(cv, matched, mock_llm)

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("Prompt" in m for m in debug_messages)
    assert any("Raw" in m or "response" in m.lower() for m in debug_messages)
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_critic_agent.py::test_critique_matches_logs_prompt_and_response_at_debug -v`
Expected: FAIL.

**Step 3: Modify `agents/critic_agent.py`**

Add at the top:
```python
import logging

logger = logging.getLogger(__name__)
```

In `critique_matches`, around the `llm.invoke(prompt)` call:

```python
logger.debug("Prompt to LLM (critic):\n%s", prompt)
raw_response = llm.invoke(prompt)
logger.debug("Raw LLM response (critic):\n%s", raw_response)
```

**Step 4: Run tests to verify**

Run: `pytest tests/test_critic_agent.py -v`
Expected: all pass.

**Step 5: Commit point** — `agents/critic_agent.py`, `tests/test_critic_agent.py` ready to commit.

---

## Task 7: Add DEBUG log to `tools/remotive_client.py`

**Files:**
- Modify: `tools/remotive_client.py`
- Test: `tests/test_remotive_client.py` (extend)

**Step 1: Write failing test**

```python
def test_fetch_remotive_jobs_logs_request_at_debug(caplog):
    import logging
    from tools.remotive_client import fetch_remotive_jobs

    mock_response = MagicMock()
    mock_response.json.return_value = {"jobs": []}
    mock_response.raise_for_status = MagicMock()

    with patch("tools.remotive_client.requests.get", return_value=mock_response):
        with caplog.at_level(logging.DEBUG, logger="tools.remotive_client"):
            fetch_remotive_jobs("python", limit=5)

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("remotive" in m.lower() or "GET" in m for m in debug_messages)
```

(Adjust imports — `MagicMock`, `patch` from `unittest.mock`.)

**Step 2: Run test to verify failure**

Run: `pytest tests/test_remotive_client.py::test_fetch_remotive_jobs_logs_request_at_debug -v`
Expected: FAIL.

**Step 3: Modify `tools/remotive_client.py`**

Add at the top:
```python
import logging

logger = logging.getLogger(__name__)
```

In `fetch_remotive_jobs`, before `requests.get`:

```python
def fetch_remotive_jobs(search: str, limit: int = 20) -> list[dict]:
    """Fetch jobs from Remotive API and return normalized results."""
    params = {"search": search, "limit": limit}
    logger.debug("GET %s params=%s", _REMOTIVE_API_URL, params)
    response = requests.get(_REMOTIVE_API_URL, params=params)
    response.raise_for_status()
    raw_jobs = response.json().get("jobs", [])
    return [normalize_remotive_job(job) for job in raw_jobs]
```

**Step 4: Run tests**

Run: `pytest tests/test_remotive_client.py -v`
Expected: all pass.

**Step 5: Commit point** — `tools/remotive_client.py`, `tests/test_remotive_client.py` ready to commit.

---

## Task 8: Add `remotive_search` field to `JobSearchState`

**Files:**
- Modify: `graph/state.py`
- Modify: `tests/test_nodes.py` (the `_make_state` helper needs the new field)

**Step 1: Modify `graph/state.py`**

Add the new field:

```python
from typing import TypedDict


class JobSearchState(TypedDict):
    cv_path: str
    cv_text: str
    cv_embedding: list
    retrieved_jobs: list
    matched_jobs: list
    critic_feedback: str
    validated_jobs: list
    report_path: str
    remotive_search: str  # mutable per-run search query for Remotive API
```

**Step 2: Update `_make_state` helper in `tests/test_nodes.py`**

Find:
```python
defaults = {
    "cv_path": "", "cv_text": "", "cv_embedding": [],
    "retrieved_jobs": [], "matched_jobs": [], "critic_feedback": "",
    "validated_jobs": [], "report_path": "",
}
```

Add `"remotive_search": ""` to defaults.

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass (this change is purely additive — no behavior changes yet).

**Step 4: Commit point** — `graph/state.py`, `tests/test_nodes.py` ready to commit.

---

## Task 9: `fetch_remote_jobs` reads `remotive_search` from state

**Files:**
- Modify: `graph/nodes.py` (`fetch_remote_jobs` function)
- Modify: `tests/test_nodes.py` (update existing or add test)

**Step 1: Write failing test**

In `tests/test_nodes.py`, add:

```python
def test_fetch_remote_jobs_uses_state_search():
    from graph.nodes import fetch_remote_jobs
    state = _make_state(remotive_search="ML engineer")
    with patch("graph.nodes.fetch_remotive_jobs", return_value=[]) as mock_fetch:
        fetch_remote_jobs(state)
    mock_fetch.assert_called_once()
    args, kwargs = mock_fetch.call_args
    # First positional arg or 'search' kwarg should be "ML engineer"
    assert args[0] == "ML engineer" or kwargs.get("search") == "ML engineer"
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_nodes.py::test_fetch_remote_jobs_uses_state_search -v`
Expected: FAIL — current implementation uses `REMOTIVE_SEARCH` from config, not state.

**Step 3: Modify `graph/nodes.py:fetch_remote_jobs`**

Replace:
```python
@log_step("Fetch jobs from Remotive")
def fetch_remote_jobs(state: JobSearchState) -> dict:
    jobs = fetch_remotive_jobs(REMOTIVE_SEARCH, limit=REMOTIVE_LIMIT)
    logger.info("Found %d remote jobs", len(jobs))
    return {"retrieved_jobs": jobs}
```

With:
```python
@log_step("Fetch jobs from Remotive")
def fetch_remote_jobs(state: JobSearchState) -> dict:
    search = state["remotive_search"]
    jobs = fetch_remotive_jobs(search, limit=REMOTIVE_LIMIT)
    logger.info("Found %d remote jobs for query '%s'", len(jobs), search)
    return {"retrieved_jobs": jobs}
```

(Remove `REMOTIVE_SEARCH` from the `from config.settings import` line if no longer used elsewhere in this module — it likely isn't.)

**Step 4: Run tests to verify**

Run: `pytest tests/test_nodes.py -v`
Expected: all pass.

**Step 5: Commit point** — `graph/nodes.py`, `tests/test_nodes.py` ready to commit.

---

## Task 10: Conditional `interrupt_before` in graph_builder

**Files:**
- Modify: `graph/graph_builder.py`
- Modify: `tests/test_graph_builder.py` (extend)

**Step 1: Write failing tests**

Append to `tests/test_graph_builder.py`:

```python
class TestInterrupts:
    @patch("graph.graph_builder.JOB_SOURCE", "remotive")
    def test_remotive_source_interrupts_before_fetch_and_report(self):
        from graph.graph_builder import build_graph
        graph = build_graph()
        # Inspect compiled graph's interrupt config
        interrupts = graph.builder.interrupt_before_nodes  # internal attr; see fallback below
        assert "fetch_remote_jobs" in interrupts
        assert "generate_report" in interrupts

    @patch("graph.graph_builder.JOB_SOURCE", "local")
    def test_local_source_only_interrupts_before_report(self):
        from graph.graph_builder import build_graph
        graph = build_graph()
        interrupts = graph.builder.interrupt_before_nodes
        assert "fetch_remote_jobs" not in interrupts
        assert "generate_report" in interrupts
```

**Note:** The exact attribute (`graph.builder.interrupt_before_nodes`) depends on the LangGraph version. If it doesn't exist, use a behavioral test instead — invoke the graph with a stub state and assert it pauses (`graph.get_state(config).next == ("fetch_remote_jobs",)`).

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_graph_builder.py -v`
Expected: FAIL — current `interrupt_before` is hardcoded to `["generate_report"]`.

**Step 3: Modify `graph/graph_builder.py`**

Replace the compile block:

```python
    interrupts = ["generate_report"]
    if JOB_SOURCE == "remotive":
        interrupts.insert(0, "fetch_remote_jobs")

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupts,
    )
    return graph
```

**Step 4: Run tests**

Run: `pytest tests/test_graph_builder.py -v`
Expected: all pass.

**Step 5: Commit point** — `graph/graph_builder.py`, `tests/test_graph_builder.py` ready to commit.

---

## Task 11: HITL handler + dispatch loop in `main.py`

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py` (new file)

**Context:** `main()` currently treats all interrupts as the "review results before report" pause. After Task 10 there are two pause points — we need to dispatch on `state.next`. We also need `handle_remotive_confirmation` for the new pause, and to wire `search_override` into the initial state.

**Step 1: Write failing tests for `handle_remotive_confirmation`**

Create `tests/test_main.py`:

```python
import builtins
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_graph():
    """A graph stub with the methods main.py uses."""
    graph = MagicMock()
    return graph


def test_handle_remotive_confirmation_yes_resumes(mock_graph, monkeypatch, capsys):
    from main import handle_remotive_confirmation

    state = MagicMock()
    state.values = {"remotive_search": "python"}

    monkeypatch.setattr(builtins, "input", lambda _: "yes")

    handle_remotive_confirmation(mock_graph, state, config={"configurable": {"thread_id": "t"}})

    mock_graph.invoke.assert_called_once_with(None, {"configurable": {"thread_id": "t"}})


def test_handle_remotive_confirmation_no_exits(mock_graph, monkeypatch):
    from main import handle_remotive_confirmation

    state = MagicMock()
    state.values = {"remotive_search": "python"}

    monkeypatch.setattr(builtins, "input", lambda _: "no")

    with pytest.raises(SystemExit):
        handle_remotive_confirmation(mock_graph, state, config={"configurable": {"thread_id": "t"}})


def test_handle_remotive_confirmation_edit_then_yes(mock_graph, monkeypatch):
    from main import handle_remotive_confirmation

    state = MagicMock()
    state.values = {"remotive_search": "python"}

    inputs = iter(["AI engineer", "yes"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    handle_remotive_confirmation(mock_graph, state, config={"configurable": {"thread_id": "t"}})

    # update_state was called with the new search
    update_calls = mock_graph.update_state.call_args_list
    assert any("AI engineer" in str(call) for call in update_calls)
    mock_graph.invoke.assert_called_once()
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `handle_remotive_confirmation` doesn't exist.

**Step 3: Modify `main.py`**

Add the new function (near `display_results`):

```python
def handle_remotive_confirmation(graph, state, config: dict) -> None:
    """Pause point before fetch_remote_jobs. Show preview, ask for confirmation or edit.

    yes  → resume the graph
    no   → sys.exit(0)
    other → treat as new search query, update state, re-prompt
    """
    while True:
        current_search = state.values.get("remotive_search", "")
        print("\n" + "=" * 60)
        print("Ready to fetch jobs from Remotive API")
        print("=" * 60)
        print("URL:    https://remotive.com/api/remote-jobs")
        print(f'Search: "{current_search}"')
        print(f"Limit:  {os.getenv('REMOTIVE_LIMIT', '20')}")
        print()

        response = input("Approve and fetch? (yes / no / type new search query):\n> ").strip()

        if response.lower() == "yes":
            graph.invoke(None, config)
            return
        if response.lower() == "no":
            print("Aborted by user.")
            sys.exit(0)
        if not response:
            continue  # empty input, re-prompt

        # Treat as new search query
        graph.update_state(config, {"remotive_search": response})
        state = graph.get_state(config)
        print(f'\nUpdated search query to: "{response}"')
```

Refactor the existing main loop (lines 40-61) to dispatch on `state.next`:

```python
def main(cv_path: str, search_override: str | None = None) -> None:
    from config.settings import REMOTIVE_SEARCH

    graph = build_graph()
    config = {"configurable": {"thread_id": "job-search-1"}}

    initial_state: JobSearchState = {
        "cv_path": cv_path,
        "cv_text": "",
        "cv_embedding": [],
        "retrieved_jobs": [],
        "matched_jobs": [],
        "critic_feedback": "",
        "validated_jobs": [],
        "report_path": "",
        "remotive_search": search_override or REMOTIVE_SEARCH,
    }

    print("Starting job search pipeline...")

    for _ in graph.stream(initial_state, config, stream_mode="values"):
        pass

    while True:
        state = graph.get_state(config)
        next_nodes = state.next

        if not next_nodes:
            break  # graph finished

        if next_nodes == ("fetch_remote_jobs",):
            handle_remotive_confirmation(graph, state, config)
            # After resume, continue streaming
            for _ in graph.stream(None, config, stream_mode="values"):
                pass
            continue

        if next_nodes == ("generate_report",):
            # Existing review loop (extracted into its own helper for clarity)
            if not handle_results_review(graph, state, config):
                # User said "no" — restart matching
                continue
            break  # accepted; graph.invoke(None) was called inside helper

        # Unknown pause point — defensive
        print(f"Unexpected pause at: {next_nodes}. Resuming.")
        graph.invoke(None, config)


def handle_results_review(graph, state, config: dict) -> bool:
    """Show validated jobs, ask user to accept. Returns True if accepted."""
    validated_jobs = state.values.get("validated_jobs", [])
    critic_feedback = state.values.get("critic_feedback", "")

    if critic_feedback:
        print(f"\nCritic says: {critic_feedback}")

    display_results(validated_jobs)

    decision = input("\nAccept these results and generate report? (yes/no): ").strip().lower()

    if decision == "yes":
        graph.invoke(None, config)
        final_state = graph.get_state(config)
        print(f"\nDone! Report saved to: {final_state.values.get('report_path', 'outputs/')}")
        return True

    print("Restarting matching with fresh analysis...")
    graph.update_state(config, {"matched_jobs": [], "validated_jobs": []}, as_node="retrieve_jobs")
    for _ in graph.stream(None, config, stream_mode="values"):
        pass
    return False
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

**Step 5: Commit point** — `main.py`, `tests/test_main.py` ready to commit.

---

## Task 12: Add markdown link to report

**Files:**
- Modify: `utils/report_generator.py`
- Modify: `tests/test_report_generator.py`

**Step 1: Write failing test**

Append to `tests/test_report_generator.py`:

```python
def test_generate_report_includes_markdown_link_when_url_present(tmp_path):
    jobs = [{
        "id": "1", "positionName": "Dev", "company": "Acme",
        "location": "Remote", "salary": "N/A", "llm_score": 7,
        "vector_score": 0.8, "url": "https://example.com/job/1",
    }]
    path = generate_report(jobs, output_dir=str(tmp_path))
    content = open(path).read()
    assert "[→ See the offer](https://example.com/job/1)" in content


def test_generate_report_falls_back_to_external_apply_link(tmp_path):
    jobs = [{
        "id": "1", "positionName": "Dev", "company": "Acme",
        "location": "Remote", "salary": "N/A", "llm_score": 7,
        "vector_score": 0.8, "externalApplyLink": "https://apply.example.com/1",
    }]
    path = generate_report(jobs, output_dir=str(tmp_path))
    content = open(path).read()
    assert "[→ See the offer](https://apply.example.com/1)" in content


def test_generate_report_omits_link_when_no_url(tmp_path):
    jobs = [{
        "id": "1", "positionName": "Dev", "company": "Acme",
        "location": "Remote", "salary": "N/A", "llm_score": 7,
        "vector_score": 0.8,
    }]
    path = generate_report(jobs, output_dir=str(tmp_path))
    content = open(path).read()
    assert "See the offer" not in content
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_report_generator.py -v`
Expected: 3 new tests FAIL.

**Step 3: Modify `utils/report_generator.py`**

Inside the `for rank, job in enumerate(jobs, start=1):` loop, after the location/salary line and before the matching skills logic, add:

```python
url = job.get("url") or job.get("externalApplyLink", "")
if url:
    lines.append(f"[→ See the offer]({url})")
```

**Step 4: Run tests**

Run: `pytest tests/test_report_generator.py -v`
Expected: all pass (existing + 3 new).

**Step 5: Commit point** — `utils/report_generator.py`, `tests/test_report_generator.py` ready to commit.

---

## Task 13: End-to-end manual smoke test

**Goal:** Run the full pipeline against the live Remotive API to validate verbose logging, HITL flow, and report URL output integration.

**Prerequisites:**
- Ollama running locally with the configured model (`OLLAMA_MODEL` from `.env`)
- A test CV PDF on disk
- `JOB_SOURCE=remotive` in `.env`
- Internet access (Remotive is a public API)

**Step 1: Smoke test — verbose mode + HITL**

Run: `python main.py path/to/cv.pdf -v -s "junior python developer"`

Expected behavior:
1. Pipeline starts; INFO logs show `→ Parse CV started`, `✓ Parse CV finished`
2. Before fetch, you see the preview:
   ```
   ===========================================
   Ready to fetch jobs from Remotive API
   ===========================================
   URL:    https://remotive.com/api/remote-jobs
   Search: "junior python developer"
   Limit:  20

   Approve and fetch? (yes / no / type new search query):
   ```
3. Type a different query (e.g., `python AI`) → preview re-shows with new query
4. Type `yes` → DEBUG logs show `GET https://remotive.com/api/remote-jobs params=...`
5. Matching agent runs with DEBUG logs showing prompts and raw LLM responses for each job
6. Review screen appears with results
7. Accept → report generated

**Step 2: Inspect the report**

Open the generated `outputs/report_*.md`.
Expected: each job has a `[→ See the offer](https://...)` clickable link below the salary line.

**Step 3: Quiet mode test**

Run: `python main.py path/to/cv.pdf` (no `-v`)
Expected:
- INFO-level logs only (no LLM prompts in terminal)
- Same HITL behavior at fetch and review
- Report still has links

**Step 4: Local mode regression check**

Set `JOB_SOURCE=local` in `.env` (or temporarily).
Run: `python main.py path/to/cv.pdf -v`
Expected:
- No fetch HITL pause (local mode skips that interrupt)
- Embed and retrieve_jobs nodes log normally
- Review HITL still works
- Report has links from local dataset's `url` field

**Step 5: Final commit point** — at this point all tasks should be committed. Run `git status` to confirm a clean tree.

---

## Final test sweep

Run: `pytest tests/ -v`
Expected: all tests pass, no skips, no warnings about deprecated logging API.

---

## Summary of files touched

| File | Change |
|---|---|
| `utils/logging_config.py` | NEW — `setup_logging(verbose)` |
| `utils/decorators.py` | NEW — `@log_step` |
| `tests/test_logging_config.py` | NEW |
| `tests/test_decorators.py` | NEW |
| `tests/test_main.py` | NEW |
| `main.py` | argparse, `setup_logging`, dispatch loop, `handle_remotive_confirmation`, `handle_results_review` extracted |
| `graph/state.py` | `+remotive_search: str` |
| `graph/graph_builder.py` | conditional `interrupt_before` list |
| `graph/nodes.py` | `@log_step` on all nodes, `print()` → `logger`, read `remotive_search` from state |
| `agents/matching_agent.py` | `logger.debug()` for prompts/responses |
| `agents/critic_agent.py` | `logger.debug()` for prompts/responses |
| `tools/remotive_client.py` | `logger.debug()` for HTTP request |
| `utils/report_generator.py` | markdown link `[→ See the offer](url)` |
| `tests/test_nodes.py` | `_make_state` updated with `remotive_search`, new test for state-driven search |
| `tests/test_graph_builder.py` | tests for conditional interrupts |
| `tests/test_matching_agent.py` | test for prompt/response DEBUG logging |
| `tests/test_critic_agent.py` | test for prompt/response DEBUG logging |
| `tests/test_remotive_client.py` | test for request DEBUG logging |
| `tests/test_report_generator.py` | 3 tests for link presence/fallback/absence |
