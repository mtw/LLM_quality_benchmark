import csv
from pathlib import Path

from llm_quality_benchmark.summary import summarize
from llm_quality_benchmark.types import ScoreRecord


def test_summarize_writes_csvs_and_handles_missing_timing(tmp_path: Path) -> None:
    records = [
        ScoreRecord(
            model="fast",
            prompt_file="p1.md",
            task_type="general",
            run_seconds=10.0,
            output_chars=100,
            instruction_following=3,
            correctness=3,
            completeness=3,
            clarity=3,
            actionability=3,
            hallucination_risk=3,
            overall=3,
            verdict="acceptable",
            pass_fail="pass",
            notes="ok",
        ),
        ScoreRecord(
            model="fast",
            prompt_file="p2.md",
            task_type="general",
            run_seconds=12.0,
            output_chars=100,
            instruction_following=3,
            correctness=3,
            completeness=3,
            clarity=3,
            actionability=3,
            hallucination_risk=3,
            overall=3,
            verdict="acceptable",
            pass_fail="pass",
            notes="ok",
        ),
        ScoreRecord(
            model="unknown_speed",
            prompt_file="p1.md",
            task_type="general",
            run_seconds=None,
            output_chars=100,
            instruction_following=5,
            correctness=5,
            completeness=5,
            clarity=5,
            actionability=5,
            hallucination_risk=5,
            overall=5,
            verdict="excellent",
            pass_fail="pass",
            notes="great",
        ),
    ]

    out = tmp_path
    summarize(records, out)

    summary_path = out / "summary.csv"
    ranked_path = out / "ranked_models.csv"
    assert summary_path.exists()
    assert ranked_path.exists()

    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[2]["run_seconds"] == ""

    with ranked_path.open(newline="", encoding="utf-8") as f:
        ranked = list(csv.DictReader(f))
    assert {r["model"] for r in ranked} == {"fast", "unknown_speed"}
    row = next(r for r in ranked if r["model"] == "unknown_speed")
    assert row["speed_score"] == "3.0"
