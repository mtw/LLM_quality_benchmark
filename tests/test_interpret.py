from pathlib import Path

from llm_quality_benchmark.interpret import format_text_report, interpret_run
from llm_quality_benchmark.summary import summarize
from llm_quality_benchmark.types import ScoreRecord


def test_interpret_run_reads_summary_and_produces_report(tmp_path: Path) -> None:
    records = [
        ScoreRecord(
            model="a",
            prompt_file="01.md",
            task_type="coding",
            run_seconds=10.0,
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
            notes="",
        ),
        ScoreRecord(
            model="a",
            prompt_file="02.md",
            task_type="general",
            run_seconds=12.0,
            output_chars=120,
            instruction_following=5,
            correctness=5,
            completeness=5,
            clarity=5,
            actionability=5,
            hallucination_risk=5,
            overall=5,
            verdict="excellent",
            pass_fail="pass",
            notes="",
        ),
        ScoreRecord(
            model="b",
            prompt_file="01.md",
            task_type="coding",
            run_seconds=9.0,
            output_chars=100,
            instruction_following=4,
            correctness=4,
            completeness=4,
            clarity=4,
            actionability=4,
            hallucination_risk=4,
            overall=4,
            verdict="acceptable",
            pass_fail="pass",
            notes="",
        ),
        ScoreRecord(
            model="b",
            prompt_file="02.md",
            task_type="general",
            run_seconds=11.0,
            output_chars=120,
            instruction_following=4,
            correctness=4,
            completeness=4,
            clarity=4,
            actionability=4,
            hallucination_risk=4,
            overall=4,
            verdict="needs_work",
            pass_fail="fail",
            notes="",
        ),
    ]

    summarize(records, tmp_path)
    result = interpret_run(tmp_path, top_n=1)
    assert result.n_records == 4
    assert result.n_models == 2
    assert result.n_prompts == 2
    assert result.top_models[0]["model"] == "a"
    assert any(f["model"] == "b" and f["prompt_file"] == "02.md" for f in result.failures)
    assert any(r["task_type"] == "coding" and r["model"] == "a" for r in result.best_by_task_type)

    text = format_text_report(result)
    assert "Top models" in text
    assert "Failures:" in text

