# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `interpret` / `report` command to summarize an existing run directory (reads `summary.csv`) into a quick text/JSON report.
- Round-robin judging mode (`rr`) that rotates judge models from a YAML/JSON config and produces per-judge, consensus, and judge-agreement CSV outputs.
- Optional progress bars for round-robin runs via `.[progress]` (uses `tqdm` when available).

## [0.1.0] - 2026-04-09

### Added
- CLI to benchmark one or more Ollama models against Markdown prompt files.
- Judge-model scoring with a fixed rubric and JSON validation.
- CSV summaries (`summary.csv`) and ranked model table (`ranked_models.csv`).
- GitHub Actions CI running `pytest` on Python 3.10–3.14.
