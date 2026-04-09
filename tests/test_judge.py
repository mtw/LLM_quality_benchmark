from pathlib import Path

import pytest

from llm_quality_benchmark.judge import (
    detect_task_type,
    extract_json,
    parse_prompt_front_matter,
    validate_score,
)


def test_parse_prompt_front_matter_empty() -> None:
    assert parse_prompt_front_matter("hello") == {}


def test_parse_prompt_front_matter_basic() -> None:
    text = "---\n# comment\nTask_Type: coding\n---\nrest"
    assert parse_prompt_front_matter(text).get("task_type") == "coding"


def test_detect_task_type_front_matter_overrides() -> None:
    prompt_text = "---\ntask_type: coding\n---\nPlease summarize this paper."
    t = detect_task_type(Path("03_summary_paper.md"), prompt_text)
    assert t == "coding"


def test_extract_json_direct() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_embedded() -> None:
    data = extract_json('note:\n{"a": 1, "b": 2}\nthanks')
    assert data == {"a": 1, "b": 2}


def test_validate_score_accepts_valid() -> None:
    valid = {
        "instruction_following": 3,
        "correctness": 4,
        "completeness": 2,
        "clarity": 5,
        "actionability": 3,
        "hallucination_risk": 4,
        "overall": 3,
        "verdict": "acceptable",
        "pass_fail": "pass",
        "notes": "ok",
    }
    assert validate_score(valid).to_dict() == valid


def test_validate_score_rejects_out_of_range() -> None:
    bad = {
        "instruction_following": 0,
        "correctness": 4,
        "completeness": 2,
        "clarity": 5,
        "actionability": 3,
        "hallucination_risk": 4,
        "overall": 3,
        "verdict": "acceptable",
        "pass_fail": "pass",
        "notes": "ok",
    }
    with pytest.raises(ValueError):
        validate_score(bad)
