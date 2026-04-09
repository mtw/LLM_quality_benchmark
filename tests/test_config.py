import json
from pathlib import Path

import pytest

from llm_quality_benchmark.config import load_benchmark_config, parse_benchmark_config


def test_parse_benchmark_config_defaults_judges_to_models() -> None:
    cfg = parse_benchmark_config({"models": ["a", "b"]})
    assert cfg.models == ["a", "b"]
    assert cfg.judges == ["a", "b"]
    assert cfg.exclude_self_judging is True


def test_parse_benchmark_config_validates_models() -> None:
    with pytest.raises(ValueError):
        parse_benchmark_config({})


def test_load_benchmark_config_json(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"models": ["a"], "judges": ["b"], "exclude_self_judging": False}), encoding="utf-8")
    cfg = load_benchmark_config(p)
    assert cfg.models == ["a"]
    assert cfg.judges == ["b"]
    assert cfg.exclude_self_judging is False

