from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    models: list[str]
    judges: list[str]
    exclude_self_judging: bool = True


def _as_list_of_str(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} must contain only non-empty strings")
        out.append(item.strip())
    return out


def parse_benchmark_config(data: dict[str, Any]) -> BenchmarkConfig:
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping/object at the top level")

    models = _as_list_of_str(data.get("models"), field="models")
    if not models:
        raise ValueError("Config must define at least one model in `models`")

    judges = _as_list_of_str(data.get("judges"), field="judges") or list(models)

    exclude_self = data.get("exclude_self_judging", True)
    if not isinstance(exclude_self, bool):
        raise ValueError("exclude_self_judging must be boolean if provided")

    return BenchmarkConfig(models=models, judges=judges, exclude_self_judging=exclude_self)


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(raw)
        return parse_benchmark_config(data)

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "YAML config requires PyYAML. Install with: python3 -m pip install -e '.[yaml]'"
            ) from exc

        data = yaml.safe_load(raw)
        return parse_benchmark_config(data)

    raise ValueError(f"Unsupported config format '{suffix}'. Use .yml/.yaml or .json")

