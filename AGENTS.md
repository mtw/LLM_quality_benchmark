# Agent Notes (LLM_Quality_Benchmark)

## What This Repo Is
- A small Python CLI that benchmarks local Ollama models against Markdown prompts, then uses a judge model to score outputs.

## How To Run
- Benchmark runner: `python3 llm_quality_benchmark.py --prompts-dir prompts --output-dir benchmark_runs --judge-model <judge> --model <model> --model <model>`
- Tests (preferred): `python3 -m pytest`

## Conventions
- Keep the CLI thin (`llm_quality_benchmark.py`) and put logic in modules.
- Tests should use `pytest` style (plain `assert`, `pytest.raises`).
- Avoid adding heavy dependencies unless needed; this is intended to stay lightweight.

## Repo Hygiene
- Prefer keeping generated artifacts out of commits (see `.gitignore`).
- If changing scoring or outputs, update `CHANGELOG.md` and relevant docs in `README.md`.
