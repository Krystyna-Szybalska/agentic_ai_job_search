import os
from datetime import datetime


def generate_report(
    jobs: list[dict],
    critic_feedback: str = "",
    cv_path: str = "",
    output_dir: str = "outputs/",
) -> str:
    """Generate a markdown report and save to output_dir. Returns file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"report_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    cv_name = os.path.basename(cv_path) if cv_path else "N/A"
    lines = [
        "# Job Match Report",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        f"\n_CV: {cv_name}_\n",
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
