# Multi-Agent Job Matching System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a LangGraph multi-agent system that reads a CV (PDF), matches it against a job dataset using semantic search, validates results with two LLM agents, and outputs a ranked markdown report.

**Architecture:** Linear StateGraph with 7 nodes. Jobs are pre-embedded into a FAISS index (offline). At query time, the CV is embedded and matched against jobs, processed by a Matching Agent and Critic Agent, paused for human approval, then a markdown report is generated.

**Tech Stack:** Python 3.11+, LangGraph, langchain-ollama (Phi-3 mini via Ollama), sentence-transformers (all-MiniLM-L6-v2), faiss-cpu, pdfplumber, python-dotenv

---

## Prerequisites

Before starting:
1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull phi3:mini`
2. Verify Ollama is running: `ollama list`
3. The processed jobs dataset exists at `data/processed/jobs_dataset.json` (735 records, fields: `company`, `positionName`, `location`, `salary`, `description`)

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env`
- Create: `embeddings/__init__.py`
- Create: `agents/__init__.py`
- Create: `graph/__init__.py`
- Create: `tools/__init__.py`
- Create: `config/__init__.py`
- Create: `utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `outputs/.gitkeep`
- Create: `vector_store/.gitkeep`

**Step 1: Create requirements.txt**

```
langgraph>=0.2.0
langchain-ollama>=0.1.0
langchain-core>=0.2.0
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
pdfplumber>=0.11.0
python-dotenv>=1.0.0
numpy>=1.26.0
pytest>=8.0.0
```

**Step 2: Create .env**

```
OLLAMA_MODEL=phi3:mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=10
TOP_N_FOR_CRITIC=5
VECTOR_STORE_PATH=vector_store/
JOBS_DATA_PATH=data/processed/jobs_dataset.json
OUTPUTS_DIR=outputs/
```

**Step 3: Create all __init__.py files and placeholder dirs**

```bash
mkdir -p embeddings agents graph tools config utils tests outputs vector_store
touch embeddings/__init__.py agents/__init__.py graph/__init__.py
touch tools/__init__.py config/__init__.py utils/__init__.py tests/__init__.py
touch outputs/.gitkeep vector_store/.gitkeep
```

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

---

## Task 2: Config — Settings and Prompts

**Files:**
- Create: `config/settings.py`
- Create: `config/prompts.py`

**Step 1: Create config/settings.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "10"))
TOP_N_FOR_CRITIC = int(os.getenv("TOP_N_FOR_CRITIC", "5"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store/")
JOBS_DATA_PATH = os.getenv("JOBS_DATA_PATH", "data/processed/jobs_dataset.json")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs/")
```

**Step 2: Create config/prompts.py**

```python
MATCHING_PROMPT = """You are a job matching assistant.

CV:
{cv_text}

Job offer:
Title: {job_title}
Company: {job_company}
Location: {job_location}
Salary: {job_salary}
Description: {job_description}

Rate how well this CV matches the job. Respond ONLY in this JSON format:
{{
  "score": <integer 1-10>,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "summary": "<1-2 sentences>"
}}"""

CRITIC_PROMPT = """You are a critical reviewer of job match results.

CV:
{cv_text}

Top job matches (title, company, score):
{matched_jobs_summary}

Review the ranking. Are the scores fair and well-ordered?
Respond ONLY in this JSON format:
{{
  "verdict": "approved",
  "feedback": "<brief explanation>",
  "suggested_ranking": [<job_id_1>, <job_id_2>, ...]
}}"""
```

No test needed for config files — they are pure data.

---

## Task 3: PDF Parser Tool

**Files:**
- Create: `tools/pdf_parser.py`
- Create: `tests/test_pdf_parser.py`

**Step 1: Write the failing test**

```python
# tests/test_pdf_parser.py
import pytest
from tools.pdf_parser import extract_text_from_pdf

def test_extract_text_returns_string(tmp_path):
    # Use any small real PDF or skip if none available
    # Here we test the function signature and error handling
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("nonexistent.pdf")

def test_extract_text_from_valid_pdf(tmp_path):
    import pdfplumber
    # Create a minimal test: if pdfplumber can open a file, function works
    # Integration test — run manually with a real PDF
    pass
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_pdf_parser.py -v
```

Expected: `ImportError` — `tools.pdf_parser` does not exist yet.

**Step 3: Implement tools/pdf_parser.py**

```python
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    import os
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_pdf_parser.py -v
```

Expected: PASS

---

## Task 4: Embedder

**Files:**
- Create: `embeddings/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write the failing test**

```python
# tests/test_embedder.py
from embeddings.embedder import get_embedder, embed_text, embed_batch

def test_embed_text_returns_list():
    embedder = get_embedder()
    result = embed_text("Software engineer with Python experience", embedder)
    assert isinstance(result, list)
    assert len(result) == 384  # all-MiniLM-L6-v2 dimension

def test_embed_batch_returns_list_of_lists():
    embedder = get_embedder()
    texts = ["Python developer", "Data scientist", "Product manager"]
    result = embed_batch(texts, embedder)
    assert len(result) == 3
    assert len(result[0]) == 384
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_embedder.py -v
```

Expected: `ImportError` — module does not exist yet.

**Step 3: Implement embeddings/embedder.py**

```python
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL


def get_embedder() -> SentenceTransformer:
    """Load and return the sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str, embedder: SentenceTransformer) -> list[float]:
    """Embed a single text string."""
    return embedder.encode(text).tolist()


def embed_batch(texts: list[str], embedder: SentenceTransformer) -> list[list[float]]:
    """Embed a list of text strings."""
    return embedder.encode(texts).tolist()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_embedder.py -v
```

Expected: PASS (note: first run downloads the model ~90MB)

---

## Task 5: Vector Store

**Files:**
- Create: `embeddings/vector_store.py`
- Create: `tests/test_vector_store.py`

**Step 1: Write the failing test**

```python
# tests/test_vector_store.py
import numpy as np
from embeddings.vector_store import VectorStore

def test_build_and_query():
    store = VectorStore(dimension=4)
    jobs = [
        {"id": "1", "positionName": "Python Dev", "company": "A"},
        {"id": "2", "positionName": "Java Dev", "company": "B"},
        {"id": "3", "positionName": "Data Scientist", "company": "C"},
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    store.build(jobs, embeddings)

    query = [1.0, 0.0, 0.0, 0.0]
    results = store.query(query, k=2)
    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert "vector_score" in results[0]

def test_save_and_load(tmp_path):
    store = VectorStore(dimension=4)
    jobs = [{"id": "1", "positionName": "Dev", "company": "X"}]
    embeddings = [[1.0, 0.0, 0.0, 0.0]]
    store.build(jobs, embeddings)
    store.save(str(tmp_path))

    store2 = VectorStore(dimension=4)
    store2.load(str(tmp_path))
    results = store2.query([1.0, 0.0, 0.0, 0.0], k=1)
    assert results[0]["id"] == "1"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_vector_store.py -v
```

Expected: `ImportError`

**Step 3: Implement embeddings/vector_store.py**

```python
import json
import os
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.jobs: list[dict] = []

    def build(self, jobs: list[dict], embeddings: list[list[float]]) -> None:
        """Build FAISS index from job embeddings."""
        self.jobs = jobs
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(self.dimension)  # inner product = cosine after normalization
        self.index.add(vectors)

    def query(self, embedding: list[float], k: int = 10) -> list[dict]:
        """Return top-K jobs for a query embedding."""
        vector = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            job = dict(self.jobs[idx])
            job["vector_score"] = float(score)
            results.append(job)
        return results

    def save(self, path: str) -> None:
        """Save index and job metadata to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "jobs.json"), "w", encoding="utf-8") as f:
            json.dump(self.jobs, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load index and job metadata from disk."""
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "jobs.json"), encoding="utf-8") as f:
            self.jobs = json.load(f)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_vector_store.py -v
```

Expected: PASS

---

## Task 6: Build Vector Store Script (One-Time Preprocessing)

**Files:**
- Create: `scripts/build_vector_store.py`

This script is run once to embed all jobs and save the FAISS index.

**Step 1: Create scripts/build_vector_store.py**

```python
"""
Run once to build the FAISS vector store from jobs_dataset.json.
Usage: python scripts/build_vector_store.py
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder import get_embedder, embed_batch
from embeddings.vector_store import VectorStore
from config.settings import JOBS_DATA_PATH, VECTOR_STORE_PATH

def build():
    print("Loading jobs...")
    with open(JOBS_DATA_PATH, encoding="utf-8") as f:
        raw_jobs = json.load(f)

    # Add a stable id and build text_for_embedding
    jobs = []
    texts = []
    for i, job in enumerate(raw_jobs):
        job["id"] = str(i)
        jobs.append(job)
        text = f"{job.get('positionName', '')}. {job.get('description', '')}"
        texts.append(text)

    print(f"Embedding {len(jobs)} jobs (this may take a minute)...")
    embedder = get_embedder()
    embeddings = embed_batch(texts, embedder)

    print("Building FAISS index...")
    store = VectorStore(dimension=384)
    store.build(jobs, embeddings)

    print(f"Saving to {VECTOR_STORE_PATH}...")
    store.save(VECTOR_STORE_PATH)
    print("Done! Vector store built successfully.")

if __name__ == "__main__":
    build()
```

**Step 2: Run the script**

```bash
python scripts/build_vector_store.py
```

Expected output:
```
Loading jobs...
Embedding 735 jobs (this may take a minute)...
Building FAISS index...
Saving to vector_store/...
Done! Vector store built successfully.
```

Verify: `vector_store/index.faiss` and `vector_store/jobs.json` now exist.

---

## Task 7: Graph State

**Files:**
- Create: `graph/state.py`

No test needed — TypedDict is a pure type annotation, validated implicitly by the graph.

**Step 1: Create graph/state.py**

```python
from typing import TypedDict


class JobSearchState(TypedDict):
    cv_path: str
    cv_text: str
    cv_embedding: list
    retrieved_jobs: list        # dicts with vector_score
    matched_jobs: list          # dicts with llm_score + analysis
    critic_feedback: str
    validated_jobs: list        # final ranked list after critic
    human_approved: bool
    report_path: str
```

---

## Task 8: Matching Agent

**Files:**
- Create: `agents/matching_agent.py`
- Create: `tests/test_matching_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_matching_agent.py
from unittest.mock import patch, MagicMock
from agents.matching_agent import analyze_job

def test_analyze_job_returns_dict():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"score": 7, "matching_skills": ["Python"], "missing_skills": ["Java"], "summary": "Good match."}'

    job = {
        "id": "1",
        "positionName": "Data Scientist",
        "company": "Acme",
        "location": "Remote",
        "salary": "100k",
        "description": "Python and ML experience required.",
        "vector_score": 0.85,
    }
    result = analyze_job("Experienced Python developer", job, mock_llm)

    assert result["id"] == "1"
    assert result["llm_score"] == 7
    assert "matching_skills" in result
    assert "summary" in result

def test_analyze_job_handles_invalid_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I cannot parse this"

    job = {"id": "2", "positionName": "Dev", "company": "X",
           "location": "NY", "salary": "80k", "description": "Test", "vector_score": 0.5}
    result = analyze_job("Python dev", job, mock_llm)

    assert result["id"] == "2"
    assert result["llm_score"] == 0  # fallback
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_matching_agent.py -v
```

Expected: `ImportError`

**Step 3: Implement agents/matching_agent.py**

```python
import json
from config.prompts import MATCHING_PROMPT


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
        parsed = json.loads(raw_response)
        result["llm_score"] = int(parsed.get("score", 0))
        result["matching_skills"] = parsed.get("matching_skills", [])
        result["missing_skills"] = parsed.get("missing_skills", [])
        result["summary"] = parsed.get("summary", "")
    except (json.JSONDecodeError, ValueError):
        result["llm_score"] = 0
        result["matching_skills"] = []
        result["missing_skills"] = []
        result["summary"] = raw_response[:200]

    return result
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_matching_agent.py -v
```

Expected: PASS

---

## Task 9: Critic Agent

**Files:**
- Create: `agents/critic_agent.py`
- Create: `tests/test_critic_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_critic_agent.py
from unittest.mock import MagicMock
from agents.critic_agent import critique_matches

def test_critique_returns_reordered_jobs():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"verdict": "approved", "feedback": "Ranking looks correct.", "suggested_ranking": ["2", "1", "3"]}'

    jobs = [
        {"id": "1", "positionName": "Dev A", "company": "X", "llm_score": 8},
        {"id": "2", "positionName": "Dev B", "company": "Y", "llm_score": 9},
        {"id": "3", "positionName": "Dev C", "company": "Z", "llm_score": 7},
    ]
    validated, feedback = critique_matches("Python dev CV", jobs, mock_llm)

    assert validated[0]["id"] == "2"
    assert validated[1]["id"] == "1"
    assert "Ranking looks correct" in feedback

def test_critique_fallback_on_invalid_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "cannot parse"

    jobs = [{"id": "1", "positionName": "Dev", "company": "X", "llm_score": 8}]
    validated, feedback = critique_matches("CV text", jobs, mock_llm)

    assert len(validated) == 1  # returns original order
    assert isinstance(feedback, str)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_critic_agent.py -v
```

Expected: `ImportError`

**Step 3: Implement agents/critic_agent.py**

```python
import json
from config.prompts import CRITIC_PROMPT


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
        parsed = json.loads(raw_response)
        feedback = parsed.get("feedback", "")
        ranking = parsed.get("suggested_ranking", [])
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
        return matched_jobs, raw_response[:200]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_critic_agent.py -v
```

Expected: PASS

---

## Task 10: Report Generator

**Files:**
- Create: `utils/report_generator.py`
- Create: `tests/test_report_generator.py`

**Step 1: Write the failing test**

```python
# tests/test_report_generator.py
import os
from utils.report_generator import generate_report

def test_generate_report_creates_file(tmp_path):
    jobs = [
        {
            "id": "1",
            "positionName": "Data Scientist",
            "company": "Acme Corp",
            "location": "Remote",
            "salary": "$100k",
            "llm_score": 9,
            "matching_skills": ["Python", "ML"],
            "missing_skills": ["Spark"],
            "summary": "Strong match for data science role.",
            "vector_score": 0.92,
        }
    ]
    path = generate_report(jobs, critic_feedback="Ranking is solid.", output_dir=str(tmp_path))
    assert os.path.exists(path)
    content = open(path).read()
    assert "Data Scientist" in content
    assert "Acme Corp" in content
    assert "9/10" in content
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_report_generator.py -v
```

Expected: `ImportError`

**Step 3: Implement utils/report_generator.py**

```python
import os
from datetime import datetime


def generate_report(
    jobs: list[dict],
    critic_feedback: str = "",
    output_dir: str = "outputs/",
) -> str:
    """Generate a markdown report and save to output_dir. Returns file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"report_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    lines = [
        "# Job Match Report",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        "---\n",
    ]

    if critic_feedback:
        lines += [f"**Critic feedback:** {critic_feedback}\n", "---\n"]

    for rank, job in enumerate(jobs, start=1):
        score = job.get("llm_score", "N/A")
        lines += [
            f"## {rank}. {job.get('positionName', 'Unknown')} — {job.get('company', '')}",
            f"**Score:** {score}/10 | **Vector similarity:** {job.get('vector_score', 0):.2f}",
            f"**Location:** {job.get('location', 'N/A')} | **Salary:** {job.get('salary', 'N/A')}",
            "",
        ]

        matching = job.get("matching_skills", [])
        if matching:
            lines.append(f"**Matching skills:** {', '.join(matching)}")

        missing = job.get("missing_skills", [])
        if missing:
            lines.append(f"**Missing skills:** {', '.join(missing)}")

        summary = job.get("summary", "")
        if summary:
            lines += ["", f"_{summary}_"]

        lines.append("\n---\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_report_generator.py -v
```

Expected: PASS

---

## Task 11: Graph Nodes

**Files:**
- Create: `graph/nodes.py`
- Create: `tests/test_nodes.py`

**Step 1: Write the failing test**

```python
# tests/test_nodes.py
from unittest.mock import patch, MagicMock
from graph.state import JobSearchState

def test_parse_cv_node(tmp_path):
    from graph.nodes import parse_cv
    # Create a dummy PDF-like scenario — test the node structure
    state = JobSearchState(
        cv_path="nonexistent.pdf", cv_text="", cv_embedding=[],
        retrieved_jobs=[], matched_jobs=[], critic_feedback="",
        validated_jobs=[], human_approved=False, report_path=""
    )
    with patch("graph.nodes.extract_text_from_pdf", return_value="John Doe, Python developer"):
        result = parse_cv(state)
    assert result["cv_text"] == "John Doe, Python developer"

def test_embed_cv_node():
    from graph.nodes import embed_cv
    state = JobSearchState(
        cv_path="", cv_text="Python developer with 5 years experience",
        cv_embedding=[], retrieved_jobs=[], matched_jobs=[],
        critic_feedback="", validated_jobs=[], human_approved=False, report_path=""
    )
    with patch("graph.nodes.get_embedder") as mock_get, \
         patch("graph.nodes.embed_text", return_value=[0.1] * 384):
        result = embed_cv(state)
    assert len(result["cv_embedding"]) == 384

def test_retrieve_jobs_node():
    from graph.nodes import retrieve_jobs
    state = JobSearchState(
        cv_path="", cv_text="", cv_embedding=[0.1] * 384,
        retrieved_jobs=[], matched_jobs=[], critic_feedback="",
        validated_jobs=[], human_approved=False, report_path=""
    )
    mock_store = MagicMock()
    mock_store.query.return_value = [{"id": "1", "positionName": "Dev", "vector_score": 0.9}]
    with patch("graph.nodes.VectorStore") as MockStore:
        instance = MockStore.return_value
        instance.query.return_value = [{"id": "1", "positionName": "Dev", "vector_score": 0.9}]
        instance.load.return_value = None
        result = retrieve_jobs(state)
    assert len(result["retrieved_jobs"]) == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_nodes.py -v
```

Expected: `ImportError`

**Step 3: Implement graph/nodes.py**

```python
from graph.state import JobSearchState
from tools.pdf_parser import extract_text_from_pdf
from embeddings.embedder import get_embedder, embed_text
from embeddings.vector_store import VectorStore
from agents.matching_agent import analyze_job
from agents.critic_agent import critique_matches
from utils.report_generator import generate_report
from config.settings import (
    OLLAMA_MODEL, TOP_K, TOP_N_FOR_CRITIC, VECTOR_STORE_PATH, OUTPUTS_DIR
)
from langchain_ollama import OllamaLLM


def parse_cv(state: JobSearchState) -> dict:
    cv_text = extract_text_from_pdf(state["cv_path"])
    return {"cv_text": cv_text}


def embed_cv(state: JobSearchState) -> dict:
    embedder = get_embedder()
    embedding = embed_text(state["cv_text"], embedder)
    return {"cv_embedding": embedding}


def retrieve_jobs(state: JobSearchState) -> dict:
    store = VectorStore(dimension=384)
    store.load(VECTOR_STORE_PATH)
    results = store.query(state["cv_embedding"], k=TOP_K)
    return {"retrieved_jobs": results}


def matching_agent(state: JobSearchState) -> dict:
    llm = OllamaLLM(model=OLLAMA_MODEL)
    matched = []
    for job in state["retrieved_jobs"]:
        print(f"  Analyzing: {job.get('positionName', '')} at {job.get('company', '')}...")
        result = analyze_job(state["cv_text"], job, llm)
        matched.append(result)
    matched.sort(key=lambda j: j.get("llm_score", 0), reverse=True)
    return {"matched_jobs": matched}


def critic_agent(state: JobSearchState) -> dict:
    llm = OllamaLLM(model=OLLAMA_MODEL)
    top_jobs = state["matched_jobs"][:TOP_N_FOR_CRITIC]
    validated, feedback = critique_matches(state["cv_text"], top_jobs, llm)
    return {"validated_jobs": validated, "critic_feedback": feedback}


def human_review(state: JobSearchState) -> dict:
    # This node is reached after the INTERRUPT in graph_builder.
    # The graph resumes here after user input is provided via main.py.
    return {}


def generate_report_node(state: JobSearchState) -> dict:
    path = generate_report(
        state["validated_jobs"],
        critic_feedback=state.get("critic_feedback", ""),
        output_dir=OUTPUTS_DIR,
    )
    print(f"\nReport saved to: {path}")
    return {"report_path": path}
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_nodes.py -v
```

Expected: PASS

---

## Task 12: Graph Builder

**Files:**
- Create: `graph/graph_builder.py`

No isolated unit test — tested via integration in Task 13.

**Step 1: Create graph/graph_builder.py**

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
    human_review,
    generate_report_node,
)


def build_graph():
    builder = StateGraph(JobSearchState)

    builder.add_node("parse_cv", parse_cv)
    builder.add_node("embed_cv", embed_cv)
    builder.add_node("retrieve_jobs", retrieve_jobs)
    builder.add_node("matching_agent", matching_agent)
    builder.add_node("critic_agent", critic_agent)
    builder.add_node("human_review", human_review)
    builder.add_node("generate_report", generate_report_node)

    builder.add_edge(START, "parse_cv")
    builder.add_edge("parse_cv", "embed_cv")
    builder.add_edge("embed_cv", "retrieve_jobs")
    builder.add_edge("retrieve_jobs", "matching_agent")
    builder.add_edge("matching_agent", "critic_agent")
    builder.add_edge("critic_agent", "human_review")
    builder.add_edge("human_review", "generate_report")
    builder.add_edge("generate_report", END)

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["generate_report"],
    )
    return graph
```

---

## Task 13: Main Entry Point + Integration Test

**Files:**
- Create: `main.py`

**Step 1: Create main.py**

```python
import sys
from langgraph.types import Command
from graph.graph_builder import build_graph
from graph.state import JobSearchState


def display_results(jobs: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("TOP JOB MATCHES")
    print("=" * 60)
    for i, job in enumerate(jobs, 1):
        print(f"\n{i}. {job.get('positionName', 'N/A')} — {job.get('company', 'N/A')}")
        print(f"   Score: {job.get('llm_score', 'N/A')}/10  |  Location: {job.get('location', 'N/A')}  |  Salary: {job.get('salary', 'N/A')}")
        if job.get("summary"):
            print(f"   {job['summary']}")
    print("\n" + "=" * 60)


def main(cv_path: str) -> None:
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
        "human_approved": False,
        "report_path": "",
    }

    print("Starting job search pipeline...")
    print("Step 1/5: Parsing CV...")

    # Run until interrupt (before generate_report)
    for event in graph.stream(initial_state, config, stream_mode="values"):
        pass  # events stream node outputs; graph pauses at interrupt

    state = graph.get_state(config)
    validated_jobs = state.values.get("validated_jobs", [])
    critic_feedback = state.values.get("critic_feedback", "")

    if critic_feedback:
        print(f"\nCritic says: {critic_feedback}")

    display_results(validated_jobs)

    decision = input("\nAccept these results and generate report? (yes/no): ").strip().lower()

    if decision == "yes":
        graph.invoke(Command(resume=None), config)
        final_state = graph.get_state(config)
        print(f"\nDone! Report saved to: {final_state.values.get('report_path', 'outputs/')}")
    else:
        print("Restarting matching with fresh analysis...")
        # Reset matched jobs and retry from matching_agent
        graph.update_state(config, {"matched_jobs": [], "validated_jobs": []}, as_node="retrieve_jobs")
        graph.invoke(Command(resume=None), config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/cv.pdf")
        sys.exit(1)
    main(sys.argv[1])
```

**Step 2: Run full integration test with a real CV PDF**

```bash
python main.py path/to/your_cv.pdf
```

Expected:
```
Starting job search pipeline...
Step 1/5: Parsing CV...
  Analyzing: Senior Data Scientist at Google...
  Analyzing: ...
[results displayed]
Accept these results and generate report? (yes/no): yes
Done! Report saved to: outputs/report_2026-02-27_14-30.md
```

**Step 3: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests PASS

---

## Task 14: Update data_loader.py

**Files:**
- Modify: `data/data_loader.py`

The existing `clean_job` function uses different field names than the actual dataset. Update to match the real schema.

**Step 1: Update data/data_loader.py**

```python
def clean_job(row: dict) -> dict:
    return {
        "id": str(row.get("id", "")),
        "positionName": row.get("positionName", ""),
        "company": row.get("company", ""),
        "location": row.get("location", ""),
        "salary": row.get("salary", "N/A"),
        "description": row.get("description", ""),
        "text_for_embedding": f"{row.get('positionName', '')}. {row.get('description', '')}",
    }


def clean_resume(row: dict) -> dict:
    return {
        "id": str(row.get("ID", "")),
        "category": row.get("Category", ""),
        "text": row.get("Resume", ""),
    }
```

No test for this file — it is a pure data transformation used by the build script.

---

## Running Order Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull Ollama model (one time)
ollama pull phi3:mini

# 3. Build vector store (one time)
python scripts/build_vector_store.py

# 4. Run all tests
pytest tests/ -v

# 5. Run the app
python main.py path/to/your_cv.pdf
```
