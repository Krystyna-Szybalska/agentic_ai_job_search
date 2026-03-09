# Agentic AI Job Search

A multi-agent job matching system built with LangGraph. Reads your CV (PDF), matches it against a dataset of job offers using semantic search, validates and reranks results with two LLM agents, and generates a ranked markdown report — with a human-in-the-loop approval step before the final output.

Built as a learning project to practice agentic architecture with real-world components: vector search, local LLMs, stateful graphs, and human interrupts.

---

## How It Works

The system runs as a linear **LangGraph StateGraph** with 7 nodes:

```
CV (PDF)
   │
   ▼
[parse_cv] ──► [embed_cv] ──► [retrieve_jobs] ──► [matching_agent] ──► [critic_agent]
                                                                               │
                                                                               ▼
                                                                        [human_review]  ◄── YOU APPROVE HERE
                                                                               │
                                                                               ▼
                                                                       [generate_report]
                                                                               │
                                                                               ▼
                                                                        report_YYYY-MM-DD.md
```

### Pipeline Steps

| Step | Node | What happens |
|------|------|-------------|
| 1 | `parse_cv` | Extracts text from your CV PDF using pdfplumber |
| 2 | `embed_cv` | Embeds the CV text with `all-MiniLM-L6-v2` |
| 3 | `retrieve_jobs` | Queries FAISS index → returns top-10 semantically similar jobs |
| 4 | `matching_agent` | Phi-3 mini scores each job 1–10, identifies matching/missing skills |
| 5 | `critic_agent` | Second LLM reviews the ranking and may reorder it |
| 6 | `human_review` | **Graph pauses** — you see the results and decide whether to accept |
| 7 | `generate_report` | Generates a ranked markdown report and saves it to `outputs/` |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Local LLM | [Ollama](https://ollama.com) — Phi-3 mini (`phi3:mini`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) — cosine similarity via IndexFlatIP |
| PDF parsing | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| Configuration | python-dotenv |
| Testing | pytest |
| Language | Python 3.11+ |

---

## Project Structure

```
agentic_ai_job_search/
│
├── main.py                         # Entry point
├── requirements.txt
├── .env.example                    # Configuration template (copy to .env)
│
├── config/
│   ├── settings.py                 # Loads env vars with defaults
│   └── prompts.py                  # LLM prompt templates
│
├── agents/
│   ├── matching_agent.py           # Scores individual job vs CV
│   └── critic_agent.py             # Reviews and reranks top matches
│
├── embeddings/
│   ├── embedder.py                 # Sentence transformer wrapper
│   └── vector_store.py             # FAISS index (build, query, save, load)
│
├── graph/
│   ├── state.py                    # JobSearchState TypedDict
│   ├── nodes.py                    # All 7 graph node functions
│   └── graph_builder.py            # StateGraph assembly + human interrupt
│
├── tools/
│   └── pdf_parser.py               # PDF text extraction
│
├── utils/
│   └── report_generator.py         # Markdown report writer
│
├── scripts/
│   └── build_vector_store.py       # One-time job embedding script
│
├── data/
│   └── processed/                  # jobs_dataset.json (not tracked — see Setup)
│
├── vector_store/                   # FAISS index files (generated, not tracked)
│
├── outputs/                        # Generated reports (generated, not tracked)
│
└── tests/
    ├── test_pdf_parser.py
    ├── test_embedder.py
    ├── test_vector_store.py
    ├── test_matching_agent.py
    ├── test_critic_agent.py
    ├── test_report_generator.py
    └── test_nodes.py
```

---

## Prerequisites

**1. Python 3.11+**

**2. Ollama** — local LLM runtime

Install from [ollama.com](https://ollama.com), then pull the model:
```bash
ollama pull phi3:mini
```

Verify it's running:
```bash
ollama list
```

**3. A CV in PDF format** — you'll need this to run the pipeline.

---

## Setup

**1. Clone the repository and create a virtual environment:**

```bash
git clone <repo-url>
cd agentic_ai_job_search
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables** (optional — defaults work out of the box):

```bash
cp .env.example .env
```

Edit `.env` if you want to change model names, paths, or search parameters. See `.env.example` for available options.

**4. Download the datasets:**

Data files are not included in the repository. This project uses two Kaggle datasets:

- **Jobs dataset (required):** [700+ Jobs Data of AI EDA 2025](https://www.kaggle.com/code/priti567/700-jobs-data-of-ai-eda-2025) — download `jobs_dataset.json` and place it in `data/processed/`.
- **Resume dataset (optional, for testing):** [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) — place PDF files in `data/raw/resume dataset/`

**5. Build the vector store** (one-time, ~1 minute):

```bash
python scripts/build_vector_store.py
```

This embeds all job offers and saves the FAISS index to `vector_store/`.

---

## Running the App

```bash
python main.py path/to/your_cv.pdf
```

**Example session:**

```
Starting job search pipeline...
Step 1/5: Parsing CV...
  Analyzing: Data Scientist at Google...
  Analyzing: ML Engineer at Spotify...
  ...

Critic says: Ranking looks appropriate given the CV's strong Python and ML background.

============================================================
TOP JOB MATCHES
============================================================

1. Senior Data Scientist — Google
   Score: 9/10  |  Location: Warsaw  |  Salary: 25000 PLN
   Strong alignment with ML and Python experience.

2. Machine Learning Engineer — Spotify
   Score: 8/10  |  Location: Remote  |  Salary: 22000 PLN
   Good match, missing Spark experience.

...

============================================================

Accept these results and generate report? (yes/no): yes

Done! Report saved to: outputs/report_2026-02-27_14-30.md
```

If you answer **no**, the pipeline reruns the matching and critic agents with a fresh analysis pass.

---

## Output

The report is a markdown file saved to `outputs/`. Example structure:

```markdown
# Job Match Report

_Generated: 2026-02-27 14:30_

**Critic feedback:** Ranking looks appropriate given the CV's strong Python background.

---

## 1. Senior Data Scientist — Google
**Score:** 9/10 | **Vector similarity:** 0.91
**Location:** Warsaw | **Salary:** 25000 PLN

**Matching skills:** Python, TensorFlow, SQL
**Missing skills:** Spark

_Strong alignment with ML and Python experience. Candidate has relevant background._

---
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 14 tests cover: PDF parsing, embedding, vector store (build/query/save/load), matching agent (including JSON fallback), critic agent (reranking + fallback), report generation, and graph nodes (mocked).

---

## Design Decisions

**Why FAISS + sentence-transformers instead of a managed vector DB?**
Keeps the stack fully local and offline. No API keys, no network dependency, instant startup.

**Why two LLM agents instead of one?**
The matching agent scores each job independently, which can lead to inconsistent relative rankings. The critic agent sees all results together and corrects ordering — a simple form of multi-agent self-review.

**Why `interrupt_before=["generate_report"]`?**
LangGraph's human-in-the-loop mechanism lets the graph pause at a checkpoint without losing state. The user reviews results in-memory and either approves or triggers a retry — without restarting the whole pipeline from scratch.

**Why Phi-3 mini?**
Small enough to run on a laptop CPU via Ollama, but capable of producing structured JSON output reliably enough for this use case.

---

## Limitations

- The jobs dataset is not included in the repository — see Setup for download instructions. To use your own dataset, replace `data/processed/jobs_dataset.json` and rerun `build_vector_store.py`.
- Phi-3 mini occasionally returns malformed JSON. Both agents handle this gracefully with fallbacks (score 0 / original ranking preserved).
- The pipeline is sequential — scoring 10 jobs takes ~2–5 minutes on CPU depending on hardware.
- `human_approved` field in state is tracked but not currently used for conditional branching — it's a hook for future extension.
