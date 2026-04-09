import csv
import json
from pathlib import Path

import pytest

import llm_quality_benchmark.round_robin as rr
from llm_quality_benchmark.config import BenchmarkConfig


def _valid_judge_json(score: int, *, pass_fail: str = "pass") -> str:
    data = {
        "instruction_following": score,
        "correctness": score,
        "completeness": score,
        "clarity": score,
        "actionability": score,
        "hallucination_risk": score,
        "overall": score,
        "verdict": "excellent" if score >= 5 else "acceptable",
        "pass_fail": pass_fail,
        "notes": "ok",
    }
    return json.dumps(data)


def test_round_robin_writes_consensus_and_agreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "01_coding.md").write_text("Write a function.", encoding="utf-8")
    (prompts_dir / "02_general.md").write_text("Say hi.", encoding="utf-8")

    out_dir = tmp_path / "run"
    cfg = BenchmarkConfig(models=["a", "b"], judges=["a", "b"], exclude_self_judging=True)

    calls: list[tuple[str, str]] = []

    def fake_run_ollama(*, model: str, prompt: str, **_kwargs) -> str:
        calls.append((model, prompt[:40]))
        # Phase A: candidate generation
        if prompt in {"Write a function.", "Say hi."}:
            return f"output from {model}"
        # Phase B: judging - return JSON-only
        if model == "a":
            return _valid_judge_json(5)
        return _valid_judge_json(3, pass_fail="fail")

    monkeypatch.setattr(rr, "run_ollama", fake_run_ollama)

    rc = rr.RoundRobinRunConfig(
        prompts_dir=prompts_dir,
        output_dir=out_dir,
        config=cfg,
        skip_existing=False,
        ollama_url=None,
        http_stream=False,
        http_retries=0,
    )
    assert rr.run_round_robin(rc) == 0

    rr_dir = out_dir / "round_robin"
    assert (rr_dir / "summary.csv").exists()
    assert (rr_dir / "consensus_ranked_models.csv").exists()
    assert (rr_dir / "judge_agreement.csv").exists()

    with (rr_dir / "consensus_ranked_models.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models = [r["model"] for r in rows]
    assert set(models) == {"a", "b"}

    # Judge 'a' scores 'b'; judge 'b' scores 'a' (self-judging excluded).
    a_row = next(r for r in rows if r["model"] == "a")
    b_row = next(r for r in rows if r["model"] == "b")
    assert a_row["n_judges"] == "1"
    assert b_row["n_judges"] == "1"


def test_round_robin_handles_invalid_judge_json_with_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "01.md").write_text("Say hi.", encoding="utf-8")

    out_dir = tmp_path / "run"
    # Use a single candidate model so we can assert exact retry count deterministically.
    cfg = BenchmarkConfig(models=["b"], judges=["a"], exclude_self_judging=True)

    attempts = 0

    def fake_run_ollama(*, model: str, prompt: str, **_kwargs) -> str:
        nonlocal attempts
        if prompt == "Say hi.":
            return f"output from {model}"
        attempts += 1
        if attempts == 1:
            # out of range -> validate_score raises
            return _valid_judge_json(0)
        return _valid_judge_json(5)

    monkeypatch.setattr(rr, "run_ollama", fake_run_ollama)

    rc = rr.RoundRobinRunConfig(
        prompts_dir=prompts_dir,
        output_dir=out_dir,
        config=cfg,
        judge_json_retries=1,
        on_judge_error="raise",
    )
    assert rr.run_round_robin(rc) == 0
    assert attempts == 2
