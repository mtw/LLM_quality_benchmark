from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import ScoreRecord


def summarize(records: list[ScoreRecord], output_dir: Path) -> None:
    summary_csv = output_dir / "summary.csv"
    ranked_csv = output_dir / "ranked_models.csv"

    fieldnames = list(asdict(records[0]).keys()) if records else []
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            for k, v in list(row.items()):
                if v is None:
                    row[k] = ""
            writer.writerow(row)

    by_model: dict[str, list[ScoreRecord]] = {}
    for rec in records:
        by_model.setdefault(rec.model, []).append(rec)

    raw_rows: list[dict[str, Any]] = []
    for model, recs in by_model.items():
        n = len(recs)

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
        return

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
            -x["combined_score"],
            -x["quality_score"],
            -x["pass_rate"],
            x["avg_run_seconds"] if isinstance(x["avg_run_seconds"], (int, float)) else float("inf"),
        )
    )

    with ranked_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(raw_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)

