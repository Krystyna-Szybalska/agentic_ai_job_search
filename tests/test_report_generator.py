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
