# Multi-Agent Job Matching System — Design

**Date:** 2026-02-27
**Stack:** Python, LangGraph, Ollama (local LLM), FAISS, sentence-transformers
**Goal:** Learning project practicing multi-agent, tool use, state management, and human-in-the-loop patterns.

---

## Overview

A LangGraph-based agentic system that takes a user's CV (PDF) and finds the best matching job offers from a local dataset. The system uses semantic search (embeddings + FAISS) for retrieval and two LLM agents for intelligent matching and validation.

---

## Project Structure

```
agentic_ai_job_search/
├── data/
│   ├── raw/                    # Kaggle datasets
│   ├── processed/
│   └── data_loader.py
├── embeddings/
│   ├── embedder.py             # sentence-transformers logic
│   └── vector_store.py         # FAISS: build, save, query
├── agents/
│   ├── matching_agent.py       # CV vs job analysis
│   └── critic_agent.py         # validation + reranking
├── graph/
│   ├── state.py                # JobSearchState TypedDict
│   ├── nodes.py                # node functions
│   └── graph_builder.py        # graph compilation
├── tools/
│   └── pdf_parser.py           # PDF text extraction
├── config/
│   ├── settings.py             # model name, top-K, paths
│   └── prompts.py              # all prompt templates
├── utils/
│   └── report_generator.py     # markdown report generation
├── outputs/                    # generated reports
├── tests/
├── .env
├── requirements.txt
└── main.py
```

---

## State

```python
class JobSearchState(TypedDict):
    cv_path: str
    cv_text: str
    cv_embedding: list[float]
    retrieved_jobs: list[dict]   # with vector_score
    matched_jobs: list[dict]     # with llm_score + analysis
    critic_feedback: str
    validated_jobs: list[dict]   # after critic validation
    human_approved: bool
    report_path: str
```

---

## Graph — Node Flow

```
START
  │
  ▼
[parse_cv]           cv_path → cv_text
  │
  ▼
[embed_cv]           cv_text → cv_embedding
  │
  ▼
[retrieve_jobs]      cv_embedding → retrieved_jobs (top-K, with vector_score)
  │
  ▼
[matching_agent]     retrieved_jobs → matched_jobs (llm_score, analysis per job)
  │
  ▼
[critic_agent]       matched_jobs → validated_jobs + critic_feedback
  │
  ▼
[human_review]       ⏸ INTERRUPT — user approves or rejects
  │
  ├─ reject ────────► [matching_agent]  (retry loop)
  │
  ▼
[generate_report]    validated_jobs → .md file in outputs/
  │
  ▼
END
```

### Node Responsibilities

| Node | Input from State | Output to State |
|---|---|---|
| `parse_cv` | `cv_path` | `cv_text` |
| `embed_cv` | `cv_text` | `cv_embedding` |
| `retrieve_jobs` | `cv_embedding` | `retrieved_jobs` |
| `matching_agent` | `retrieved_jobs` | `matched_jobs` |
| `critic_agent` | `matched_jobs` | `validated_jobs`, `critic_feedback` |
| `human_review` | `validated_jobs` | `human_approved` |
| `generate_report` | `validated_jobs` | `report_path` |

---

## Agents

### Matching Agent

Processes each job offer individually (loop in node, not in LLM) to avoid context overflow with small models. Returns structured JSON.

**Prompt template (`config/prompts.py`):**
```
You are a job matching assistant.

CV summary:
{cv_text}

Job offer:
Title: {job_title}
Company: {job_company}
Location: {job_location}
Salary: {job_salary}
Description: {job_description}

Rate how well this CV matches the job. Respond ONLY in this JSON format:
{
  "score": <1-10>,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "summary": "<1-2 sentences>"
}
```

Note: `matching_skills` and `missing_skills` are best-effort — may be omitted in report if model output is unreliable.

### Critic Agent

Receives top-N matched jobs (e.g. top 5) to stay within context limits. Validates scores and suggests final ranking.

**Prompt template:**
```
You are a critical reviewer of job matches.

CV summary:
{cv_text}

These are the top job matches with scores:
{matched_jobs_summary}

Review the ranking. Are the scores fair?
Respond ONLY in this JSON format:
{
  "verdict": "approved" | "needs_revision",
  "feedback": "<what is wrong or why approved>",
  "suggested_ranking": [job_id1, job_id2, ...]
}
```

Both agents use `json.loads()` parsing with a fallback (keep original if parsing fails).

---

## Human-in-the-Loop

Implemented via LangGraph's `interrupt_before` mechanism with `MemorySaver` checkpointer.

```python
# graph_builder.py
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["generate_report"]
)

# main.py
result = graph.invoke(initial_state, config)
display_results(result["validated_jobs"])
decision = input("Accept results? (yes/no): ")
if decision == "yes":
    graph.invoke(Command(resume=True), config)
else:
    graph.invoke(Command(resume=False), config)  # retries from matching_agent
```

---

## Embedding Pipeline

**Preprocessing (once, offline):**
```
jobs.json → embedder.py → FAISS index (saved to vector_store/)
```

**Query time (per run):**
```
CV text → sentence-transformers → cv_embedding → FAISS query → top-K jobs
```

Model: `all-MiniLM-L6-v2` (fast, local, good semantic quality for job matching).

---

## Output Report

Markdown file saved to `outputs/report_YYYY-MM-DD_HH-MM.md` containing:
- Ranked list of top job matches
- For each job: title, company, location, salary, LLM score
- Matching/missing skills (best-effort)
- Critic agent feedback summary

---

## Tech Stack

| Component | Library | Reason |
|---|---|---|
| Agent graph | `langgraph` | project goal |
| Local LLM | `ollama` + `langchain-ollama` | 4GB VRAM, Phi-3 mini |
| Embeddings | `sentence-transformers` | local, `all-MiniLM-L6-v2` |
| Vector store | `faiss-cpu` | simple, no server needed |
| PDF parsing | `pdfplumber` | reliable for CVs |
| Config | `python-dotenv` | `.env` for paths/settings |

---

## Learning Concepts Covered

| Concept | Where |
|---|---|
| Multi-agent | `matching_agent` + `critic_agent` as separate nodes |
| Tool use | `pdf_parser`, `vector_store` as callable tools within nodes |
| State management | `JobSearchState` TypedDict flowing through all nodes |
| Human-in-the-loop | `interrupt_before=["generate_report"]` + `MemorySaver` |
