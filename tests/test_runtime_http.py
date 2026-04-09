import json

import pytest

import llm_quality_benchmark.runtime as rt


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamResponse:
    def __init__(self, chunks: list[dict]) -> None:
        self._lines = [json.dumps(c).encode("utf-8") + b"\n" for c in chunks]

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_normalize_base_url_strips_trailing_slash() -> None:
    assert rt._normalize_base_url("http://x:11434/") == "http://x:11434"


def test_run_ollama_http_uses_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        body = req.data.decode("utf-8")
        seen["payload"] = json.loads(body)
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr(rt.urllib.request, "urlopen", fake_urlopen)

    out = rt.run_ollama_http(
        base_url="http://host:11434",
        model="m",
        prompt="p",
        temperature=0.0,
        timeout=5,
    )
    assert out == "ok"
    assert seen["url"] == "http://host:11434/api/generate"
    assert seen["payload"]["model"] == "m"
    assert seen["payload"]["prompt"] == "p"
    assert seen["payload"]["stream"] is False
    assert seen["payload"]["options"]["temperature"] == 0.0


def test_run_ollama_http_passes_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_urlopen(req, timeout):
        body = req.data.decode("utf-8")
        seen["payload"] = json.loads(body)
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr(rt.urllib.request, "urlopen", fake_urlopen)
    rt.run_ollama_http(
        base_url="http://host:11434",
        model="m",
        prompt="p",
        temperature=0.2,
        options={"num_predict": 123},
        timeout=5,
    )
    assert seen["payload"]["options"]["temperature"] == 0.2
    assert seen["payload"]["options"]["num_predict"] == 123


def test_post_json_wraps_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(rt.urllib.request, "urlopen", raise_timeout)

    with pytest.raises(RuntimeError, match="timed out after 5s"):
        rt._post_json("http://host:11434/api/generate", {"model": "m"}, timeout=5)


def test_post_json_stream_wraps_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(rt.urllib.request, "urlopen", raise_timeout)

    with pytest.raises(RuntimeError, match="timed out after 5s"):
        rt._post_json_stream("http://host:11434/api/generate", {"model": "m"}, timeout=5)


def test_run_ollama_http_retries_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def raise_then_succeed(req, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("timed out")
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr(rt.urllib.request, "urlopen", raise_then_succeed)
    monkeypatch.setattr(rt.time, "sleep", lambda _: None)

    out = rt.run_ollama_http(
        base_url="http://host:11434",
        model="m",
        prompt="p",
        timeout=5,
        retries=1,
    )
    assert out == "ok"
    assert attempts == 2


def test_run_ollama_http_retries_on_timeout_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """TimeoutError from _post_json is wrapped in RuntimeError — retry logic should still catch it."""
    attempts = 0

    def raise_then_succeed(req, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("timed out")
        return _FakeStreamResponse([{"response": "ok", "done": True}])

    monkeypatch.setattr(rt.urllib.request, "urlopen", raise_then_succeed)
    monkeypatch.setattr(rt.time, "sleep", lambda _: None)

    out = rt.run_ollama_http(
        base_url="http://host:11434",
        model="m",
        prompt="p",
        timeout=5,
        retries=1,
        stream=True,
    )
    assert out == "ok"
    assert attempts == 2


def test_run_ollama_http_exhausts_retries_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_timeout(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(rt.urllib.request, "urlopen", always_timeout)
    monkeypatch.setattr(rt.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="timed out after 5s"):
        rt.run_ollama_http(
            base_url="http://host:11434",
            model="m",
            prompt="p",
            timeout=5,
            retries=2,
        )
