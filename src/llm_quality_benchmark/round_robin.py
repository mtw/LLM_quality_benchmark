from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .interpret import compute_ranked_models
from .judge import build_judge_prompt, detect_task_type, extract_json, safe_name, validate_score
from .runtime import (
    load_prompts,
    read_run_seconds,
    run_ollama,
    write_json,
    write_run_meta,
    write_text,
)
from .types import ScoreRecord, score_record_from_judge


log = logging.getLogger(__name__)


def _tqdm_if_available():
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return None
    return tqdm


@dataclass(frozen=True)
class RoundRobinRunConfig:
    prompts_dir: Path
    output_dir: Path
    config: BenchmarkConfig
    ollama_url: str | None = None
    temperature: float = 0.0
    judge_temperature: float = 0.0
    num_predict: int | None = None
    judge_num_predict: int | None = None
    timeout: int = 1800
    http_stream: bool = False
    http_retries: int = 0
    skip_existing: bool = False
    show_progress: bool = True


@dataclass(frozen=True)
class RoundRobinScoreRecord:
    judge_model: str
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

    def as_score_record(self) -> ScoreRecord:
        return ScoreRecord(
            model=self.model,
            prompt_file=self.prompt_file,
            task_type=self.task_type,
            run_seconds=self.run_seconds,
            output_chars=self.output_chars,
            instruction_following=self.instruction_following,
            correctness=self.correctness,
            completeness=self.completeness,
            clarity=self.clarity,
            actionability=self.actionability,
            hallucination_risk=self.hallucination_risk,
            overall=self.overall,
            verdict=self.verdict,
            pass_fail=self.pass_fail,
            notes=self.notes,
        )


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_dataclass_csv(path: Path, records: list[Any]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            for k, v in list(row.items()):
                if v is None:
                    row[k] = ""
            writer.writerow(row)


def _spearman_rho(order_a: list[str], order_b: list[str]) -> float | None:
    ranks_a = {m: i + 1 for i, m in enumerate(order_a)}
    ranks_b = {m: i + 1 for i, m in enumerate(order_b)}
    common = sorted(set(ranks_a) & set(ranks_b))
    n = len(common)
    if n < 2:
        return None
    ssd = 0.0
    for m in common:
        d = float(ranks_a[m] - ranks_b[m])
        ssd += d * d
    return 1.0 - (6.0 * ssd) / (n * (n * n - 1.0))


def _rank_order_from_ranked_rows(rows: list[dict[str, Any]]) -> list[str]:
    def key(row: dict[str, Any]) -> tuple[float, str]:
        cs = row.get("combined_score", 0)
        try:
            cs_f = float(cs)
        except Exception:
            cs_f = 0.0
        return (-cs_f, str(row.get("model", "")))

    return [str(r["model"]) for r in sorted(rows, key=key)]


def _rr_dir(output_dir: Path) -> Path:
    return output_dir / "round_robin"


def run_round_robin(config: RoundRobinRunConfig) -> int:
    prompts = load_prompts(config.prompts_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rr_dir = _rr_dir(config.output_dir)
    rr_dir.mkdir(parents=True, exist_ok=True)
    (rr_dir / "scores_by_judge").mkdir(parents=True, exist_ok=True)
    (rr_dir / "ranked_models_by_judge").mkdir(parents=True, exist_ok=True)
    (rr_dir / "summary_by_judge").mkdir(parents=True, exist_ok=True)

    tqdm = _tqdm_if_available() if config.show_progress else None
    use_tqdm = bool(tqdm) and sys.stderr.isatty()

    # Phase A: generate candidate outputs once.
    gen_total = len(config.config.models) * len(prompts)
    gen_skipped = 0
    gen_pbar = tqdm(total=gen_total, desc="Generate", unit="run") if use_tqdm else None
    if not use_tqdm:
        log.info("Phase A: generating outputs (%d model×prompt runs)", gen_total)

    try:
        for model in config.config.models:
            model_safe = safe_name(model)
            for prompt_path in prompts:
                prompt_stem = prompt_path.stem
                prompt_text = prompt_path.read_text(encoding="utf-8")

                raw_out_path = config.output_dir / "outputs" / model_safe / f"{prompt_stem}.txt"
                meta_json_path = config.output_dir / "meta" / model_safe / f"{prompt_stem}.json"

                if config.skip_existing and raw_out_path.exists() and meta_json_path.exists():
                    gen_skipped += 1
                    if gen_pbar:
                        gen_pbar.update(1)
                    continue

                started = time.perf_counter()
                model_options: dict[str, Any] = {}
                if config.num_predict is not None:
                    model_options["num_predict"] = int(config.num_predict)

                output_text = run_ollama(
                    model=model,
                    prompt=prompt_text,
                    temperature=config.temperature,
                    timeout=config.timeout,
                    base_url=config.ollama_url,
                    stream=bool(config.http_stream),
                    options=(model_options or None),
                    retries=int(config.http_retries),
                )
                run_seconds = time.perf_counter() - started
                write_text(raw_out_path, output_text)
                write_run_meta(
                    meta_json_path,
                    run_seconds=run_seconds,
                    temperature=config.temperature,
                    model=model,
                    prompt_file=prompt_path.name,
                )
                if gen_pbar:
                    gen_pbar.update(1)
    finally:
        if gen_pbar:
            gen_pbar.close()

    if not use_tqdm:
        log.info("Phase A complete (skipped=%d, wrote=%d).", gen_skipped, gen_total - gen_skipped)

    # Phase B: judge in round-robin.
    all_records: list[RoundRobinScoreRecord] = []
    records_by_judge: dict[str, list[RoundRobinScoreRecord]] = {}

    judge_pairs = 0
    for judge_model in config.config.judges:
        for model in config.config.models:
            if config.config.exclude_self_judging and model == judge_model:
                continue
            judge_pairs += 1
    judge_total = judge_pairs * len(prompts)
    judge_pbar = tqdm(total=judge_total, desc="Judge", unit="score") if use_tqdm else None
    if not use_tqdm:
        log.info("Phase B: judging (%d judge×model×prompt scores)", judge_total)

    judged_skipped = 0

    for judge_model in config.config.judges:
        judge_safe = safe_name(judge_model)
        for model in config.config.models:
            if config.config.exclude_self_judging and model == judge_model:
                continue

            model_safe = safe_name(model)
            for prompt_path in prompts:
                prompt_stem = prompt_path.stem
                prompt_text = prompt_path.read_text(encoding="utf-8")
                task_type = detect_task_type(prompt_path, prompt_text)

                raw_out_path = config.output_dir / "outputs" / model_safe / f"{prompt_stem}.txt"
                meta_json_path = config.output_dir / "meta" / model_safe / f"{prompt_stem}.json"
                if not raw_out_path.exists():
                    raise FileNotFoundError(f"Missing model output: {raw_out_path}")

                output_text = raw_out_path.read_text(encoding="utf-8")
                run_seconds = read_run_seconds(meta_json_path) if meta_json_path.exists() else None

                score_path = rr_dir / "scores_by_judge" / judge_safe / model_safe / f"{prompt_stem}.json"
                judge_data = None
                if config.skip_existing and score_path.exists():
                    try:
                        judge_raw = json.loads(score_path.read_text(encoding="utf-8"))
                        judge_data = validate_score(judge_raw)
                        judged_skipped += 1
                    except Exception:
                        judge_data = None

                if judge_data is None:
                    judge_prompt = build_judge_prompt(prompt_text, output_text, task_type)
                    judge_options: dict[str, Any] = {}
                    if config.judge_num_predict is not None:
                        judge_options["num_predict"] = int(config.judge_num_predict)
                    judge_raw_text = run_ollama(
                        model=judge_model,
                        prompt=judge_prompt,
                        temperature=config.judge_temperature,
                        timeout=config.timeout,
                        base_url=config.ollama_url,
                        stream=bool(config.http_stream),
                        options=(judge_options or None),
                        retries=int(config.http_retries),
                    )
                    judge_data = validate_score(extract_json(judge_raw_text))
                    write_json(score_path, judge_data.to_dict())

                score_record = score_record_from_judge(
                    model=model,
                    prompt_file=prompt_path.name,
                    task_type=task_type,
                    run_seconds=run_seconds,
                    output_text=output_text,
                    judge_data=judge_data,
                )
                rr_rec = RoundRobinScoreRecord(judge_model=judge_model, **asdict(score_record))
                all_records.append(rr_rec)
                records_by_judge.setdefault(judge_model, []).append(rr_rec)
                if judge_pbar:
                    judge_pbar.update(1)

    if judge_pbar:
        judge_pbar.close()
    elif not use_tqdm:
        log.info("Phase B complete (skipped=%d, wrote=%d).", judged_skipped, judge_total - judged_skipped)

    if not all_records:
        return 1

    # Prompt-level round robin summary.
    _write_dataclass_csv(rr_dir / "summary.csv", all_records)

    # Ranked models per judge.
    ranked_by_judge: dict[str, list[dict[str, Any]]] = {}
    for judge_model, recs in records_by_judge.items():
        judge_safe = safe_name(judge_model)
        score_recs = [r.as_score_record() for r in recs]
        ranked = compute_ranked_models(score_recs)
        ranked_by_judge[judge_model] = ranked
        _write_dict_csv(rr_dir / "ranked_models_by_judge" / f"{judge_safe}.csv", ranked)
        _write_dataclass_csv(rr_dir / "summary_by_judge" / f"{judge_safe}.csv", recs)

    # Consensus aggregation across judges (combined_score + robustness).
    consensus_rows: list[dict[str, Any]] = []
    per_model_scores: dict[str, list[float]] = {}
    per_model_judges: dict[str, list[str]] = {}
    for judge_model, rows in ranked_by_judge.items():
        for row in rows:
            model = str(row.get("model", ""))
            try:
                cs = float(row.get("combined_score", 0.0))
            except Exception:
                continue
            per_model_scores.setdefault(model, []).append(cs)
            per_model_judges.setdefault(model, []).append(judge_model)

    for model, scores in per_model_scores.items():
        scores_sorted = sorted(scores)
        consensus_rows.append(
            {
                "model": model,
                "n_judges": len(scores),
                "combined_score_mean": round(statistics.mean(scores), 3),
                "combined_score_median": round(statistics.median(scores), 3),
                "combined_score_stdev": (round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0),
                "combined_score_min": round(min(scores_sorted), 3),
                "combined_score_max": round(max(scores_sorted), 3),
                "judges": ";".join(per_model_judges.get(model, [])),
            }
        )

    consensus_rows.sort(key=lambda r: (-float(r["combined_score_median"]), -float(r["combined_score_mean"]), r["model"]))
    _write_dict_csv(rr_dir / "consensus_ranked_models.csv", consensus_rows)

    # Judge agreement (pairwise Spearman over ranked order).
    agreement_rows: list[dict[str, Any]] = []
    judges = list(ranked_by_judge.keys())
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            a = judges[i]
            b = judges[j]
            order_a = _rank_order_from_ranked_rows(ranked_by_judge[a])
            order_b = _rank_order_from_ranked_rows(ranked_by_judge[b])
            rho = _spearman_rho(order_a, order_b)
            agreement_rows.append(
                {
                    "judge_a": a,
                    "judge_b": b,
                    "n_common_models": len(set(order_a) & set(order_b)),
                    "spearman_rho": ("" if rho is None else round(float(rho), 4)),
                }
            )
    _write_dict_csv(rr_dir / "judge_agreement.csv", agreement_rows)

    # Meta
    meta = {
        "created_ts": int(time.time()),
        "prompts_dir": str(config.prompts_dir),
        "output_dir": str(config.output_dir),
        "models": list(config.config.models),
        "judges": list(config.config.judges),
        "exclude_self_judging": bool(config.config.exclude_self_judging),
    }
    (rr_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0
