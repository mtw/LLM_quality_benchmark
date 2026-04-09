from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        raise ValueError("base_url must be a non-empty string")
    return base_url.rstrip("/")


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if getattr(exc, "fp", None) else ""
        raise RuntimeError(f"Ollama HTTP error {exc.code} for {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama request timed out after {timeout}s for {url}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from Ollama at {url}:\n{raw}") from exc


def _post_json_stream(url: str, payload: dict[str, Any], timeout: int) -> str:
    """
    Ollama streaming returns newline-delimited JSON objects. We concatenate the
    incremental `response` fields until `done: true`.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            chunks: list[str] = []
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    part = obj.get("response")
                    if part:
                        chunks.append(str(part))
                    if obj.get("done") is True:
                        break
            return "".join(chunks).strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if getattr(exc, "fp", None) else ""
        raise RuntimeError(f"Ollama HTTP error {exc.code} for {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama request timed out after {timeout}s for {url}") from exc


def run_ollama_http(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = 1800,
    stream: bool = False,
    options: dict[str, Any] | None = None,
    retries: int = 0,
    retry_backoff: float = 1.0,
) -> str:
    base_url = _normalize_base_url(base_url)
    url = f"{base_url}/api/generate"
    merged_options: dict[str, Any] = {"temperature": float(temperature)}
    if options:
        merged_options.update(options)

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": bool(stream),
        "options": merged_options,
    }

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if stream:
                return _post_json_stream(url, payload, timeout=timeout)
            data = _post_json(url, payload, timeout=timeout)
            if not isinstance(data, dict) or "response" not in data:
                raise RuntimeError(f"Unexpected Ollama response shape from {url}: {data}")
            return str(data.get("response", "")).strip()
        except TimeoutError as exc:
            last_exc = exc
        except RuntimeError as exc:
            # Retry transient transport errors and timeouts from our wrappers.
            if "Failed to reach Ollama" in str(exc) or "timed out" in str(exc):
                last_exc = exc
            else:
                raise

        if attempt < retries:
            time.sleep(max(0.0, float(retry_backoff)) * (2**attempt))

    raise RuntimeError(
        f"Ollama request timed out after {timeout}s (retries={retries}) for {url} model={model}"
    ) from last_exc


def run_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = 1800,
    base_url: str | None = None,
    stream: bool = False,
    options: dict[str, Any] | None = None,
    retries: int = 0,
    retry_backoff: float = 1.0,
) -> str:
    if base_url:
        return run_ollama_http(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
            options=options,
            retries=retries,
            retry_backoff=retry_backoff,
        )

    cmd = [
        "ollama",
        "run",
        model,
        "--temperature",
        str(temperature),
    ]
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ollama run failed for model '{model}' with code {proc.returncode}\n"
            f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )
    return proc.stdout.strip()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prompts(prompts_dir: Path) -> list[Path]:
    files = sorted([p for p in prompts_dir.glob("*.md") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .md prompt files found in {prompts_dir}")
    return files


def write_run_meta(
    path: Path,
    *,
    run_seconds: float,
    temperature: float,
    model: str,
    prompt_file: str,
) -> None:
    write_json(
        path,
        {
            "run_seconds": round(run_seconds, 3),
            "temperature": temperature,
            "model": model,
            "prompt_file": prompt_file,
            "ts": int(time.time()),
        },
    )


def read_run_seconds(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sec = data.get("run_seconds")
        if isinstance(sec, (int, float)) and sec >= 0:
            return float(sec)
    except Exception:
        return None
    return None
