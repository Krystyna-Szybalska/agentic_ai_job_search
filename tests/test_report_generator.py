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
    content = open(path, encoding="utf-8").read()
    assert "Data Scientist" in content
    assert "Acme Corp" in content
    assert "9/10" in content


def test_generate_report_includes_markdown_link_when_url_present(tmp_path):
    jobs = [{
        "id": "1", "positionName": "Dev", "company": "Acme",
        "location": "Remote", "salary": "N/A", "llm_score": 7,
        "vector_score": 0.8, "url": "https://example.com/job/1",
    }]
    path = generate_report(jobs, output_dir=str(tmp_path))
    content = open(path, encoding="utf-8").read()
    assert "[→ See the offer](https://example.com/job/1)" in content


def test_generate_report_falls_back_to_external_apply_link(tmp_path):
    jobs = [{
        "id": "1", "positionName": "Dev", "company": "Acme",
        "location": "Remote", "salary": "N/A", "llm_score": 7,
        "vector_score": 0.8, "externalApplyLink": "https://apply.example.com/1",
    }]
    path = generate_report(jobs, output_dir=str(tmp_path))
    content = open(path, encoding="utf-8").read()
    assert "[→ See the offer](https://apply.example.com/1)" in content


def test_generate_report_omits_link_when_no_url(tmp_path):
    jobs = [{
        "id": "1", "positionName": "Dev", "company": "Acme",
        "location": "Remote", "salary": "N/A", "llm_score": 7,
        "vector_score": 0.8,
    }]
    path = generate_report(jobs, output_dir=str(tmp_path))
    content = open(path, encoding="utf-8").read()
    assert "See the offer" not in content
