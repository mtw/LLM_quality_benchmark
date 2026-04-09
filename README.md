# LLM Quality Benchmark (Ollama, Local or LAN)

This repo contains a small Python CLI that benchmarks one or more Ollama models against a directory of Markdown prompts, saves raw model outputs, and uses a separate "judge" model to score each output against a fixed rubric. It then produces CSV summaries and a ranked model table.

It is designed for:
- Comparing multiple local/LAN Ollama models on the same prompt set
- Keeping raw outputs and judge JSON for later inspection
- Getting a simple sortable `ranked_models.csv` across several quality dimensions

## What It Does

For each `--model` and each prompt file in `--prompts-dir`:
1. Runs the model on the prompt text
2. Writes the raw output to `output_dir/outputs/<model>/<prompt>.txt`
3. Builds a judge prompt (rubric + task hint + original task + candidate response)
4. Runs the judge model and parses/validates its JSON
5. Writes judge JSON to `output_dir/scores/<model>/<prompt>.json`
6. Writes run metadata (including timing) to `output_dir/meta/<model>/<prompt>.json`

Finally it writes:
- `output_dir/summary.csv`: per-prompt per-model rows
- `output_dir/ranked_models.csv`: per-model aggregates with a combined score

## Prompt Format

Put prompt files in a directory (default examples are in `prompts/`) as `*.md`.

Optional: you can declare a task type in Markdown front matter to override automatic heuristics:

```md
---
task_type: coding
---

Write a Python function that ...
```

Supported `task_type` values: `coding`, `summary`, `reasoning`, `shell`, `general`.

## Quick Start

### 1) Create and activate a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install (optional but recommended)

Editable install makes imports and the console script work consistently:

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
```

To install pytest for running tests:

```bash
python3 -m pip install -e '.[test]'
```

To enable YAML config files for round-robin mode:

```bash
python3 -m pip install -e '.[yaml]'
```

To show progress bars during round-robin runs:

```bash
python3 -m pip install -e '.[progress]'
```

### 3) Run the benchmark against an Ollama host on your LAN

If your models run on another machine in your LAN, point to its Ollama HTTP API:

```bash
python3 llm_quality_benchmark.py \
  --prompts-dir prompts \
  --output-dir benchmark_runs \
  --ollama-url http://my-llm-host:11434 \
  --judge-model llama3.1:8b \
  --model llama3.1:8b \
  --model qwen2.5-coder:7b
```

The CLI uses `POST /api/generate` with `stream: false`.

You can also set the base URL via environment variable:

```bash
export OLLAMA_BASE_URL=http://my-llm-host:11434
```

### 4) Local Ollama fallback

If `--ollama-url` is omitted, the runner uses the local `ollama` CLI from `PATH`.

```bash
python3 llm_quality_benchmark.py \
  --prompts-dir prompts \
  --output-dir benchmark_runs \
  --judge-model llama3.1:8b \
  --model llama3.1:8b
```

## Outputs

After a run completes:
- `benchmark_runs/summary.csv` contains per-prompt scores and metadata
- `benchmark_runs/ranked_models.csv` contains per-model aggregates and ranking
- `benchmark_runs/outputs/` contains raw model outputs
- `benchmark_runs/scores/` contains judge JSON outputs
- `benchmark_runs/meta/` contains run timing and config metadata

## Round-robin judging (multi-judge)

In round-robin mode, every judge model scores every candidate model (optionally excluding itself), using the same prompt set and reusing the same candidate outputs. This helps reduce single-judge bias.

Create a config file (YAML or JSON):

```yaml
models:
  - llama3.1:8b
  - qwen2.5-coder:7b
judges:
  - llama3.1:8b
  - qwen2.5-coder:7b
exclude_self_judging: true
```

Run it:

```bash
python3 llm_quality_benchmark.py rr \
  --config benchmark.yml \
  --prompts-dir prompts \
  --output-dir benchmark_runs \
  --ollama-url http://my-llm-host:11434 \
  --skip-existing
```

Key outputs are written under `benchmark_runs/round_robin/`:
- `summary.csv`: prompt-level rows with `judge_model`
- `ranked_models_by_judge/<judge>.csv`: per-judge ranking
- `consensus_ranked_models.csv`: aggregated ranking across judges (median/mean + stdev)
- `judge_agreement.csv`: pairwise judge agreement (Spearman rho over model order)

## Interpreting results

To generate a quick human-readable report from `summary.csv`:

```bash
python3 llm_quality_benchmark.py interpret --run-dir benchmark_runs --top 5
```

JSON output:

```bash
python3 llm_quality_benchmark.py interpret --run-dir benchmark_runs --format json
```

## Notes / Tips

- Use `--skip-existing` to avoid re-running prompts that already have output and score files.
- The judge is intentionally strict and expects JSON-only output. If your judge model sometimes wraps JSON in text, the parser attempts to extract the first `{...}` block.

## CI

GitHub Actions runs `pytest` on pushes and pull requests for Python 3.10 through 3.14, installing this project with `.[test]`.

## Contributing / Changelog

- Contributing guide: `CONTRIBUTING.md`
- Release notes: `CHANGELOG.md`
