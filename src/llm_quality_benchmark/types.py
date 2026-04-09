from __future__ import annotations

from dataclasses import dataclass

from .judge import JudgeScore


@dataclass
class ScoreRecord:
    model: str
    prompt_file: str
    task_type: str
    run_seconds: float | None
    output_chars: int
    instruction_following: int
    correctness: int
    completeness: int
    clarity: int
    actionability: int
    hallucination_risk: int
    overall: int
    verdict: str
    pass_fail: str
    notes: str


def score_record_from_judge(
    *,
    model: str,
    prompt_file: str,
    task_type: str,
    run_seconds: float | None,
    output_text: str,
    judge_data: JudgeScore,
) -> ScoreRecord:
    return ScoreRecord(
        model=model,
        prompt_file=prompt_file,
        task_type=task_type,
        run_seconds=run_seconds,
        output_chars=len(output_text),
        instruction_following=judge_data.instruction_following,
        correctness=judge_data.correctness,
        completeness=judge_data.completeness,
        clarity=judge_data.clarity,
        actionability=judge_data.actionability,
        hallucination_risk=judge_data.hallucination_risk,
        overall=judge_data.overall,
        verdict=judge_data.verdict,
        pass_fail=judge_data.pass_fail,
        notes=judge_data.notes,
    )
