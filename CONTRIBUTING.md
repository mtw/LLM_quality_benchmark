# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e '.[test]'
```

Run tests:

```bash
python3 -m pytest
```

## Project conventions

- Keep the top-level runner (`llm_quality_benchmark.py`) thin; put logic in `src/llm_quality_benchmark/`.
- Keep dependencies lightweight (stdlib preferred).
- Tests use `pytest` style (`assert`, `pytest.raises`).

## Adding or updating prompts

- Prompts live in `prompts/` as `*.md`.
- Optional front matter:
  - `task_type: coding|summary|reasoning|shell|general`

Try to keep prompt files stable over time so benchmark runs stay comparable.

## Running a benchmark

Example:

```bash
python3 llm_quality_benchmark.py \
  --prompts-dir prompts \
  --output-dir benchmark_runs \
  --ollama-url http://my-llm-host:11434 \
  --judge-model llama3.1:8b \
  --model llama3.1:8b \
  --model qwen2.5-coder:7b
```

## Release checklist (suggested)

- Update version in `pyproject.toml`.
- Add an entry to `CHANGELOG.md`.
- Run `python3 -m pytest`.
- Ensure generated artifacts are not committed (see `.gitignore`).

