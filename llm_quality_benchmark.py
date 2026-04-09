#!/usr/bin/env python3
"""
Local Ollama quality benchmark runner.

What it does
------------
1. Reads prompt files from a directory
2. Runs each target model on each prompt
3. Saves raw model outputs
4. Uses a judge model to score each output with a fixed rubric
5. Writes JSON score files
6. Produces a CSV summary and a ranked CSV

Recommended prompt filenames
----------------------------
prompts/
  01_coding_merge_intervals.md
  02_coding_debug_function.md
  03_summary_paper_a.md
  04_summary_paper_b.md
  05_reasoning_architecture.md
  06_reasoning_failure_modes.md
  07_shell_logs_a.md
  08_shell_logs_b.md

Example usage
-------------
python llm_quality_benchmark.py \
  --prompts-dir prompts \
  --output-dir benchmark_runs \
  --ollama-url "http://my-llm-host:11434" \
  --judge-model "glm-4.7-flash:latest" \
  --model "qwen3-coder:30b" \
  --model "glm-4.7-flash:latest" \
  --model "deepseek-r1:14b" \
  --model "gemma4:e4b"

Notes
-----
- Requires `ollama` in PATH.
- Uses temperature 0 by default for reproducibility.
- The judge is also local via Ollama.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from llm_quality_benchmark.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
