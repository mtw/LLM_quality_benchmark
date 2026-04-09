from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ScoreRecord


def _parse_int(value: str) -> int:
    v = (value or "").strip()
    if v == "":
        raise ValueError("expected int, got empty string")
    return int(v)


def _parse_float_or_none(value: str) -> float | None:
    v = (value or "").strip()
    if v == "":
        return None
    return float(v)


def load_summary_csv(path: Path) -> list[ScoreRecord]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records: list[ScoreRecord] = []
    for row in rows:
        records.append(
            ScoreRecord(
                model=str(row.get("model", "")),
                prompt_file=str(row.get("prompt_file", "")),
                task_type=str(row.get("task_type", "")),
                run_seconds=_parse_float_or_none(str(row.get("run_seconds", ""))),
                output_chars=_parse_int(str(row.get("output_chars", ""))),
                instruction_following=_parse_int(str(row.get("instruction_following", ""))),
                correctness=_parse_int(str(row.get("correctness", ""))),
                completeness=_parse_int(str(row.get("completeness", ""))),
                clarity=_parse_int(str(row.get("clarity", ""))),
                actionability=_parse_int(str(row.get("actionability", ""))),
                hallucination_risk=_parse_int(str(row.get("hallucination_risk", ""))),
                overall=_parse_int(str(row.get("overall", ""))),
                verdict=str(row.get("verdict", "")),
                pass_fail=str(row.get("pass_fail", "")),
                notes=str(row.get("notes", "")),
            )
        )
    return records


def compute_ranked_models(records: list[ScoreRecord]) -> list[dict[str, Any]]:
    by_model: dict[str, list[ScoreRecord]] = {}
    for rec in records:
        by_model.setdefault(rec.model, []).append(rec)

    raw_rows: list[dict[str, Any]] = []
    for model, recs in by_model.items():
        n = len(recs)
        if n == 0:
            continue

        def avg(attr: str) -> float:
            return sum(getattr(r, attr) for r in recs) / n

        avg_instruction = avg("instruction_following")
        avg_correctness = avg("correctness")
        avg_completeness = avg("completeness")
        avg_clarity = avg("clarity")
        avg_actionability = avg("actionability")
        avg_hallucination = avg("hallucination_risk")
        avg_overall = avg("overall")
        secs = [r.run_seconds for r in recs if r.run_seconds is not None and r.run_seconds >= 0]
        avg_run_seconds = (sum(secs) / len(secs)) if secs else None
        avg_output_chars = sum(r.output_chars for r in recs) / n
        passes = sum(1 for r in recs if r.pass_fail == "pass")
        pass_rate = passes / n

        quality_score = (
            0.15 * avg_instruction
            + 0.25 * avg_correctness
            + 0.20 * avg_completeness
            + 0.10 * avg_clarity
            + 0.10 * avg_actionability
            + 0.20 * avg_hallucination
        )

        raw_rows.append(
            {
                "model": model,
                "n_prompts": n,
                "avg_instruction_following": round(avg_instruction, 3),
                "avg_correctness": round(avg_correctness, 3),
                "avg_completeness": round(avg_completeness, 3),
                "avg_clarity": round(avg_clarity, 3),
                "avg_actionability": round(avg_actionability, 3),
                "avg_hallucination_risk": round(avg_hallucination, 3),
                "avg_overall": round(avg_overall, 3),
                "pass_rate": round(pass_rate, 3),
                "avg_run_seconds": (round(avg_run_seconds, 3) if avg_run_seconds is not None else ""),
                "avg_output_chars": round(avg_output_chars, 1),
                "quality_score_raw": quality_score,
            }
        )

    if not raw_rows:
        return []

    timing_values = [r["avg_run_seconds"] for r in raw_rows if isinstance(r["avg_run_seconds"], (int, float))]
    min_sec = min(timing_values) if timing_values else None
    max_sec = max(timing_values) if timing_values else None

    for row in raw_rows:
        avg_sec = row["avg_run_seconds"] if isinstance(row["avg_run_seconds"], (int, float)) else None
        if avg_sec is None or min_sec is None or max_sec is None:
            speed_score = 3.0
        elif max_sec == min_sec:
            speed_score = 5.0
        else:
            norm = (max_sec - avg_sec) / (max_sec - min_sec)
            speed_score = 1.0 + 4.0 * norm

        row["speed_score"] = round(speed_score, 3)
        row["quality_score"] = round(row["quality_score_raw"], 3)
        row["combined_score"] = round(0.8 * row["quality_score_raw"] + 0.2 * speed_score, 3)
        del row["quality_score_raw"]

    raw_rows.sort(
        key=lambda x: (
            -float(x["combined_score"]),
            -float(x["quality_score"]),
            -float(x["pass_rate"]),
            x["avg_run_seconds"] if isinstance(x["avg_run_seconds"], (int, float)) else float("inf"),
        )
    )
    return raw_rows


@dataclass(frozen=True)
class InterpretedRun:
    run_dir: str
    n_records: int
    n_models: int
    n_prompts: int
    top_models: list[dict[str, Any]]
    failures: list[dict[str, str]]
    best_by_task_type: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_dir": self.run_dir,
                "n_records": self.n_records,
                "n_models": self.n_models,
                "n_prompts": self.n_prompts,
                "top_models": self.top_models,
                "failures": self.failures,
                "best_by_task_type": self.best_by_task_type,
            },
            indent=2,
            ensure_ascii=False,
        )


def interpret_run(run_dir: Path, *, top_n: int = 5) -> InterpretedRun:
    run_dir = Path(run_dir)
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary.csv in {run_dir}")

    records = load_summary_csv(summary_csv)
    ranked = compute_ranked_models(records)
    top_models = ranked[: max(0, int(top_n))]

    prompt_files = sorted({r.prompt_file for r in records})
    failures = [
        {"model": r.model, "prompt_file": r.prompt_file, "task_type": r.task_type, "verdict": r.verdict}
        for r in records
        if r.pass_fail != "pass"
    ]

    by_task: dict[str, list[ScoreRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_type, []).append(r)

    best_by_task_type: list[dict[str, Any]] = []
    for task_type, task_recs in sorted(by_task.items(), key=lambda kv: kv[0]):
        per_model: dict[str, list[ScoreRecord]] = {}
        for r in task_recs:
            per_model.setdefault(r.model, []).append(r)

        best_row: dict[str, Any] | None = None
        for model, recs in per_model.items():
            n = len(recs)
            avg_overall = sum(r.overall for r in recs) / n
            pass_rate = sum(1 for r in recs if r.pass_fail == "pass") / n
            row = {"task_type": task_type, "model": model, "n_prompts": n, "avg_overall": round(avg_overall, 3), "pass_rate": round(pass_rate, 3)}
            if best_row is None or (row["avg_overall"], row["pass_rate"]) > (best_row["avg_overall"], best_row["pass_rate"]):
                best_row = row
        if best_row is not None:
            best_by_task_type.append(best_row)

    return InterpretedRun(
        run_dir=str(run_dir),
        n_records=len(records),
        n_models=len({r.model for r in records}),
        n_prompts=len(prompt_files),
        top_models=top_models,
        failures=failures,
        best_by_task_type=best_by_task_type,
    )


def format_text_report(result: InterpretedRun) -> str:
    lines: list[str] = []
    lines.append(f"Run: {result.run_dir}")
    lines.append(f"Records: {result.n_records} | Models: {result.n_models} | Prompts: {result.n_prompts}")

    if result.top_models:
        lines.append("")
        lines.append("Top models (combined_score):")
        for i, row in enumerate(result.top_models, start=1):
            model = row.get("model", "")
            cs = row.get("combined_score", "")
            qs = row.get("quality_score", "")
            ss = row.get("speed_score", "")
            pr = row.get("pass_rate", "")
            lines.append(f"{i:>2}. {model}  combined={cs}  quality={qs}  speed={ss}  pass_rate={pr}")

    if result.best_by_task_type:
        lines.append("")
        lines.append("Best by task_type (avg_overall):")
        for row in result.best_by_task_type:
            lines.append(
                f"- {row['task_type']}: {row['model']}  avg_overall={row['avg_overall']}  pass_rate={row['pass_rate']}  n={row['n_prompts']}"
            )

    if result.failures:
        lines.append("")
        lines.append("Failures:")
        for row in result.failures[:20]:
            lines.append(f"- {row['model']} :: {row['prompt_file']} ({row['task_type']}) verdict={row['verdict']}")
        if len(result.failures) > 20:
            lines.append(f"... and {len(result.failures) - 20} more")

    return "\n".join(lines).rstrip() + "\n"

