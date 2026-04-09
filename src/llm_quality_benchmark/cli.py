from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
import sys
import time
from pathlib import Path

from .judge import JudgeScore, build_judge_prompt, detect_task_type, extract_json, safe_name, validate_score
from .interpret import format_text_report, interpret_run
from .runtime import load_prompts, read_run_seconds, run_ollama, write_json, write_run_meta, write_text
from .summary import summarize
from .types import ScoreRecord, score_record_from_judge


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerConfig:
    prompts_dir: Path
    output_dir: Path
    judge_model: str
    models: list[str]
    ollama_url: str | None = None
    temperature: float = 0.0
    judge_temperature: float = 0.0
    num_predict: int | None = None
    judge_num_predict: int | None = None
    timeout: int = 1800
    http_stream: bool = False
    http_retries: int = 0
    skip_existing: bool = False


@dataclass(frozen=True)
class RunPaths:
    raw_out_path: Path
    score_json_path: Path
    meta_json_path: Path
    model_safe: str
    prompt_stem: str


@dataclass(frozen=True)
class RunLayout:
    output_dir: Path

    def paths_for(self, *, model: str, prompt_path: Path) -> RunPaths:
        model_safe = safe_name(model)
        prompt_stem = prompt_path.stem

        model_out_dir = self.output_dir / "outputs" / model_safe
        model_score_dir = self.output_dir / "scores" / model_safe
        model_meta_dir = self.output_dir / "meta" / model_safe

        return RunPaths(
            raw_out_path=model_out_dir / f"{prompt_stem}.txt",
            score_json_path=model_score_dir / f"{prompt_stem}.json",
            meta_json_path=model_meta_dir / f"{prompt_stem}.json",
            model_safe=model_safe,
            prompt_stem=prompt_stem,
        )


def run_benchmark(config: RunnerConfig) -> int:
    prompts = load_prompts(config.prompts_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    layout = RunLayout(config.output_dir)

    records: list[ScoreRecord] = []

    for model in config.models:
        for prompt_path in prompts:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            task_type = detect_task_type(prompt_path, prompt_text)
            paths = layout.paths_for(model=model, prompt_path=prompt_path)

            if config.skip_existing and paths.raw_out_path.exists() and paths.score_json_path.exists():
                try:
                    judge_raw = json.loads(paths.score_json_path.read_text(encoding="utf-8"))
                    judge_data = validate_score(judge_raw)
                    output_text = paths.raw_out_path.read_text(encoding="utf-8")
                    prior_seconds = read_run_seconds(paths.meta_json_path) if paths.meta_json_path.exists() else None
                    records.append(
                        score_record_from_judge(
                            model=model,
                            prompt_file=prompt_path.name,
                            task_type=task_type,
                            run_seconds=prior_seconds,
                            output_text=output_text,
                            judge_data=judge_data,
                        )
                    )
                    log.info("[skip] %s :: %s", model, prompt_path.name)
                    continue
                except Exception as exc:
                    log.warning(
                        "[warn] Failed to reuse existing files for %s / %s: %s",
                        model,
                        prompt_path.name,
                        exc,
                    )

            log.info("[run ] %s :: %s", model, prompt_path.name)
            started = time.perf_counter()
            model_options = {}
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
            write_text(paths.raw_out_path, output_text)
            write_run_meta(
                paths.meta_json_path,
                run_seconds=run_seconds,
                temperature=config.temperature,
                model=model,
                prompt_file=prompt_path.name,
            )

            judge_prompt = build_judge_prompt(prompt_text, output_text, task_type)

            log.info("[judge] %s <- %s :: %s", config.judge_model, model, prompt_path.name)
            judge_options = {}
            if config.judge_num_predict is not None:
                judge_options["num_predict"] = int(config.judge_num_predict)
            judge_raw = run_ollama(
                model=config.judge_model,
                prompt=judge_prompt,
                temperature=config.judge_temperature,
                timeout=config.timeout,
                base_url=config.ollama_url,
                stream=bool(config.http_stream),
                options=(judge_options or None),
                retries=int(config.http_retries),
            )
            judge_data: JudgeScore = validate_score(extract_json(judge_raw))
            write_json(paths.score_json_path, judge_data.to_dict())

            records.append(
                score_record_from_judge(
                    model=model,
                    prompt_file=prompt_path.name,
                    task_type=task_type,
                    run_seconds=round(run_seconds, 3),
                    output_text=output_text,
                    judge_data=judge_data,
                )
            )

    if not records:
        log.error("No records produced.")
        return 1

    summarize(records, config.output_dir)
    log.info("\nDone. Results written to: %s", config.output_dir)
    log.info("- %s", config.output_dir / "summary.csv")
    log.info("- %s", config.output_dir / "ranked_models.csv")
    log.info("- %s", config.output_dir / "outputs")
    log.info("- %s", config.output_dir / "scores")
    log.info("- %s", config.output_dir / "meta")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Ollama quality benchmark.")
    parser.add_argument("--prompts-dir", type=Path, required=True, help="Directory containing .md prompt files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs, scores, CSV summaries")
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL"),
        help="Base URL for Ollama HTTP API, e.g. http://host:11434 (or set OLLAMA_BASE_URL). If omitted, uses local `ollama` CLI.",
    )
    parser.add_argument("--judge-model", required=True, help="Ollama model used as judge")
    parser.add_argument(
        "--model", action="append", required=True, help="Target model to benchmark; repeat flag for multiple models"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for benchmarked models")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Temperature for judge model")
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Max tokens to generate for benchmarked models (Ollama option `num_predict`).",
    )
    parser.add_argument(
        "--judge-num-predict",
        type=int,
        default=None,
        help="Max tokens to generate for judge model (Ollama option `num_predict`).",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds per ollama run")
    parser.add_argument(
        "--http-stream",
        action="store_true",
        help="When using --ollama-url, enable streaming responses to reduce long silent waits.",
    )
    parser.add_argument(
        "--http-retries",
        type=int,
        default=0,
        help="When using --ollama-url, retry transient HTTP failures this many times.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip prompt/model runs if output and score files already exist"
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        prompts_dir=args.prompts_dir,
        output_dir=args.output_dir,
        ollama_url=args.ollama_url,
        judge_model=args.judge_model,
        models=list(args.model),
        temperature=float(args.temperature),
        judge_temperature=float(args.judge_temperature),
        num_predict=args.num_predict,
        judge_num_predict=args.judge_num_predict,
        timeout=int(args.timeout),
        http_stream=bool(args.http_stream),
        http_retries=int(args.http_retries),
        skip_existing=bool(args.skip_existing),
    )


def _main_run(argv: list[str] | None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_benchmark(_config_from_args(args))


def _build_interpret_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpret an existing benchmark run directory.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Benchmark run output directory containing summary.csv (and optionally ranked_models.csv).",
    )
    parser.add_argument("--top", type=int, default=5, help="How many top models to show.")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    return parser


def _main_interpret(argv: list[str] | None) -> int:
    parser = _build_interpret_parser()
    args = parser.parse_args(argv)
    result = interpret_run(Path(args.run_dir), top_n=int(args.top))
    if args.format == "json":
        sys.stdout.write(result.to_json() + "\n")
    else:
        sys.stdout.write(format_text_report(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if argv_list and argv_list[0] in {"interpret", "report"}:
        return _main_interpret(argv_list[1:])
    return _main_run(argv_list)
