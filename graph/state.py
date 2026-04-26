from typing import TypedDict


class JobSearchState(TypedDict):
    cv_path: str
    cv_text: str
    cv_embedding: list
    retrieved_jobs: list        # dicts with vector_score
    matched_jobs: list          # dicts with llm_score + analysis
    critic_feedback: str
    validated_jobs: list        # final ranked list after critic
    report_path: str
    remotive_search: str  # mutable per-run search query for Remotive API
