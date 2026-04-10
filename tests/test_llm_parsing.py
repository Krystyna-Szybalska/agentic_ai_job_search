from utils.llm_parsing import (
    strip_code_fences,
    parse_llm_json,
    extract_json_field,
    extract_json_number,
    extract_string_list,
    clamp_score,
    sanitize_string_list,
)


def test_strip_code_fences_closed():
    text = '```json\n{"score": 7}\n```'
    assert strip_code_fences(text) == '{"score": 7}'


def test_strip_code_fences_unclosed():
    text = '```json\n{"score": 7}'
    assert strip_code_fences(text) == '{"score": 7}'


def test_strip_code_fences_no_fences():
    text = '{"score": 7}'
    assert strip_code_fences(text) == '{"score": 7}'


def test_parse_llm_json_valid():
    result = parse_llm_json('```json\n{"score": 7}\n```')
    assert result == {"score": 7}


def test_parse_llm_json_invalid():
    assert parse_llm_json("not json at all") is None


def test_extract_json_field():
    text = '{"feedback": "Looks good", "verdict": "approved"}'
    assert extract_json_field(text, "feedback") == "Looks good"


def test_extract_json_field_missing():
    assert extract_json_field('{"other": 1}', "feedback") == ""


def test_extract_json_number():
    text = '{"score": 8, "other": "x"}'
    assert extract_json_number(text, "score") == 8


def test_extract_json_number_missing():
    assert extract_json_number('{"other": "x"}', "score") == 0


def test_extract_string_list():
    text = '{"matching_skills": ["Python", "ML", "SQL"]}'
    assert extract_string_list(text, "matching_skills") == ["Python", "ML", "SQL"]


def test_extract_string_list_truncated():
    text = '{"matching_skills": ["Python", "ML'
    assert extract_string_list(text, "matching_skills") == ["Python"]


def test_extract_string_list_missing():
    assert extract_string_list('{"other": 1}', "matching_skills") == []


def test_clamp_score_normal():
    assert clamp_score(7) == 7


def test_clamp_score_too_high():
    assert clamp_score(15) == 10


def test_clamp_score_negative():
    assert clamp_score(-3) == 0


def test_clamp_score_invalid():
    assert clamp_score("not a number") == 0


def test_sanitize_string_list_valid():
    assert sanitize_string_list(["Python", "ML"]) == ["Python", "ML"]


def test_sanitize_string_list_mixed():
    assert sanitize_string_list(["Python", 42, 3.14]) == ["Python", "42", "3.14"]


def test_sanitize_string_list_not_a_list():
    assert sanitize_string_list("Python") == []


def test_sanitize_string_list_filters_non_primitives():
    assert sanitize_string_list(["ok", {"nested": True}]) == ["ok"]
