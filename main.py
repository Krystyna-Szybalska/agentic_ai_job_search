import os
import sys
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
        graph.invoke(None, config)
        final_state = graph.get_state(config)
        print(f"\nDone! Report saved to: {final_state.values.get('report_path', 'outputs/')}")
    else:
        print("Restarting matching with fresh analysis...")
        # Reset matched jobs and retry from matching_agent
        graph.update_state(config, {"matched_jobs": [], "validated_jobs": []}, as_node="retrieve_jobs")
        graph.invoke(None, config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/cv.pdf")
        sys.exit(1)

    cv_file = sys.argv[1]
    if not os.path.isfile(cv_file):
        print(f"Error: file not found: {cv_file}")
        sys.exit(1)
    if not cv_file.lower().endswith(".pdf"):
        print("Error: file must be a PDF (.pdf)")
        sys.exit(1)

    main(cv_file)
