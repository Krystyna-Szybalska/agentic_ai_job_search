import os
import sys
import argparse
from config.settings import REMOTIVE_SEARCH, REMOTIVE_LIMIT
from graph.graph_builder import build_graph
from graph.state import JobSearchState
from utils.logging_config import setup_logging


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


def handle_remotive_confirmation(graph, state, config: dict) -> None:
    """Pause point before fetch_remote_jobs.

    Shows the request preview and accepts:
      yes  -> resume the graph
      no   -> sys.exit(0)
      other -> treat as new search query, update state, re-prompt
    """
    while True:
        current_search = state.values.get("remotive_search", "")
        print("\n" + "=" * 60)
        print("Ready to fetch jobs from Remotive API")
        print("=" * 60)
        print("URL:    https://remotive.com/api/remote-jobs")
        print(f'Search: "{current_search}"')
        print(f"Limit:  {REMOTIVE_LIMIT}")
        print()

        response = input("Approve and fetch? (yes / no / type new search query):\n> ").strip()

        if response.lower() == "yes":
            return  # dispatch loop owns resumption via graph.stream(None, ...)
        if response.lower() == "no":
            print("Aborted by user.")
            sys.exit(0)
        if not response:
            continue  # empty input, re-prompt

        # Treat as new search query
        graph.update_state(config, {"remotive_search": response})
        state = graph.get_state(config)
        print(f'\nUpdated search query to: "{response}"')


def handle_results_review(graph, state, config: dict) -> bool:
    """Show validated jobs, ask user to accept. Returns True if accepted, False if rejected.

    On rejection: restarts matching by clearing state and streaming again.
    """
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


def main(cv_path: str, search_override: str | None = None) -> None:
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
            for _ in graph.stream(None, config, stream_mode="values"):
                pass
            continue

        if next_nodes == ("generate_report",):
            if not handle_results_review(graph, state, config):
                # User said "no" -- restart matching (handle_results_review already did the streaming)
                continue
            break

        # Unknown pause point -- defensive
        print(f"Unexpected pause at: {next_nodes}. Resuming.")
        graph.invoke(None, config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def validate_cv_path(cv_path: str) -> None:
    if not os.path.isfile(cv_path):
        sys.exit(f"Error: file not found: {cv_path}")
    if not cv_path.lower().endswith(".pdf"):
        sys.exit("Error: file must be a PDF (.pdf)")


if __name__ == "__main__":
    args = parse_args()
    validate_cv_path(args.cv_path)
    setup_logging(verbose=args.verbose)
    main(args.cv_path, search_override=args.search)
