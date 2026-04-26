# Agentic AI Job Search

## What this project does

A LangGraph-based multi-agent system that matches a user's CV (PDF) against job listings using local LLM + vector search. Supports two job sources: a local FAISS-indexed dataset (~735 listings) or the Remotive API (live remote job listings). Controlled by the `JOB_SOURCE` config parameter.

**Pipeline:** `parse_cv -> (embed_cv -> retrieve_jobs | fetch_remote_jobs) -> matching_agent -> critic_agent -> (interrupt) -> generate_report`

**Entry points:**
- `python main.py path/to/cv.pdf` — run the full pipeline
- `python -m scripts.build_vector_store` — one-time FAISS index build

## Architecture

```
main.py                          CLI entry point, human-in-the-loop retry loop
config/settings.py               Environment config (dotenv + defaults), get_llm() factory
config/prompts.py                LLM prompt templates (matching + critic)
graph/state.py                   JobSearchState TypedDict (8 fields)
graph/graph_builder.py           LangGraph StateGraph assembly, MemorySaver, interrupt
graph/nodes.py                   7 node functions (thin wrappers calling tools/agents)
agents/matching_agent.py         Per-job scoring (1-10), skill extraction, summary
agents/critic_agent.py           Top-N reranking and validation
embeddings/embedder.py           SentenceTransformer wrapper (all-MiniLM-L6-v2)
embeddings/vector_store.py       FAISS IndexFlatIP (cosine via normalized IP)
tools/pdf_parser.py              pdfplumber text extraction
tools/remotive_client.py         Remotive API client (fetch + normalize jobs)
utils/llm_parsing.py             Shared LLM output parsing (JSON extraction, fallbacks)
utils/report_generator.py        Markdown report output
scripts/build_vector_store.py    Offline index builder (run via python -m scripts.build_vector_store)
```

### Key design decisions

- **Two-agent pattern:** Matching agent scores individually, critic agent validates relative ranking. This compensates for small-LLM inconsistency.
- **Graceful JSON fallback:** Shared `utils/llm_parsing.py` handles code fences, json.loads, and regex fallback. Both agents use it. Essential because small LLMs produce malformed JSON.
- **Context window management:** Truncation limits configurable via `CV_TEXT_MAX_CHARS` / `JOB_DESC_MAX_CHARS` in settings. Increase when switching to larger-context models.
- **Human-in-the-loop:** LangGraph `interrupt_before=["generate_report"]` pauses for user approval. Rejection retries matching+critic in a loop until the user accepts.
- **LLM factory:** `get_llm()` in settings.py uses `ChatOllama` (chat API) wrapped to return plain strings. This is required for thinking models like Qwen3 — the generate API doesn't support disabling thinking mode. `OLLAMA_NUM_PREDICT` defaults to 4096 to accommodate thinking tokens.
- **Dual job source:** `JOB_SOURCE=local` uses the FAISS vector store (offline). `JOB_SOURCE=remotive` fetches live jobs from the Remotive API using `REMOTIVE_SEARCH` as the query. The graph conditionally routes between the two paths at build time.
- **Local-first:** Ollama for LLM, SentenceTransformers for embeddings, FAISS for vector search. Network only needed when `JOB_SOURCE=remotive`.

## Tech stack

- Python 3.11+
- LangGraph (StateGraph, MemorySaver, interrupt)
- langchain-ollama (OllamaLLM)
- sentence-transformers (all-MiniLM-L6-v2)
- faiss-cpu (IndexFlatIP, 384 dimensions)
- pdfplumber
- requests (Remotive API)
- pytest

## Coding conventions

- **Naming:** snake_case functions/variables, PascalCase classes, ALL_CAPS constants
- **Imports:** stdlib, then third-party, then local. Use `from module import name`.
- **Type hints:** Use them on function signatures. Keep them simple (`list[dict]`, `str`, not elaborate generics).
- **Shared utilities:** Reusable logic (e.g., LLM output parsing) goes in `utils/`. Don't duplicate across agents.
- **Private helpers:** Prefix with underscore for module-internal helpers.
- **Node functions:** Thin wrappers. Accept `state: JobSearchState`, return `dict` with only the fields they update. Business logic lives in agents/tools/utils, not in nodes.
- **Error handling:** Catch specific exceptions. Prefer graceful fallback over raising. Only validate at boundaries (user input, LLM output).
- **Tests:** pytest with `unittest.mock.patch` for LLM calls. Use `tmp_path` for file I/O. Test both happy path and fallback behavior.
- **Comments:** Minimal. Docstrings on public functions (1-2 lines). Inline comments only for non-obvious logic.
- **Config:** All tunables go in `.env` with defaults in `config/settings.py`. No magic numbers in business logic.

## How to run tests

```bash
pytest tests/ -v
```

## Guiding principles for expansion

- **Keep the graph linear and simple.** Add nodes to the existing pipeline rather than introducing complex branching unless there's a clear reason. If branching is needed, document the decision.
- **Agents stay focused.** Each agent does one thing (score, critique, etc.). New capabilities = new agent, not a bigger prompt.
- **Tools are stateless functions.** PDF parser, embedder, vector store — pure functions or simple classes with no hidden state. Easy to test, easy to mock.
- **Nodes are thin.** They wire state to tools/agents. No business logic in nodes.
- **Config over code.** New tunables go to `.env` / `settings.py`, not hardcoded.
- **Test the boundaries.** LLM output parsing, file I/O, user input — these break. Internal pure functions rarely need defensive coding.
- **Don't over-abstract.** Three similar lines > premature abstraction. Add abstractions only when there are 3+ concrete use cases.
