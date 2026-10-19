import base64
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def handler(monkeypatch, tmp_path):
    # Only the external RunPod SDK and heavyweight RVC boundary are replaced.
    for name in ["runpod", "runpod.serverless", "runpod.serverless.utils",
                 "runpod.serverless.utils.rp_download", "runpod.serverless.modules",
                 "runpod.serverless.modules.rp_logger"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["runpod.serverless.utils.rp_download"].file = lambda url: {"file_path": str(tmp_path / "input.mp3")}
    sys.modules["runpod.serverless.modules.rp_logger"].RunPodLogger = lambda: logging.getLogger("handler-test")
    model_dir = tmp_path / "Singer"
    model_dir.mkdir()
    (tmp_path / "input.mp3").write_bytes(b"original audio")
    output = tmp_path / "cover.mp3"
    output.write_bytes(b"converted audio")
    pipeline = types.SimpleNamespace(rvc_models_dir=str(tmp_path), song_cover_pipeline=lambda *a, **kw: str(output))
    monkeypatch.setitem(sys.modules, "main", pipeline)
    spec = importlib.util.spec_from_file_location("handler_under_test", Path(__file__).resolve().parents[1] / "rp_handler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, pipeline, tmp_path


def test_inference_failure_keeps_original_traceback_and_job_context(handler):
    module, pipeline, folder = handler

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA out of memory: original failure")

    pipeline.song_cover_pipeline = fail
    result = module.handler({"id": "job-u1", "input": {"song_input_url": "https://files/track.mp3", "rvc_model": "Singer"}})
    assert result["error"] == "CUDA out of memory: original failure"
    assert "RuntimeError: CUDA out of memory" in result["traceback"]
    assert result["job_id"] == "job-u1"
    assert result["stage"] == "inference"
    assert result["error_type"] == "PROVIDER_ERROR"
    assert not (folder / "input.mp3").exists()


def test_successful_handler_keeps_existing_audio_contract(handler):
    module, _, folder = handler
    result = module.handler({"id": "job-u1", "input": {"song_input_url": "https://files/track.mp3", "rvc_model": "Singer"}})
    assert base64.b64decode(result["audio_base64"]) == b"converted audio"
    assert result["format"] == "mp3"
    assert result["model_used"] == "Singer"
    assert not (folder / "input.mp3").exists()


def test_download_fallback_does_not_reuse_another_jobs_file(handler, monkeypatch):
    module, _, folder = handler

    def fail(url):
        raise RuntimeError("SDK downloader unavailable")

    monkeypatch.setattr(module, "rp_file", fail)
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(folder))
    monkeypatch.setattr(module.urllib.request, "urlretrieve", lambda url, name: Path(name).write_bytes(url.encode()))
    first = module.download_audio_from_url("https://files/one.mp3")
    second = module.download_audio_from_url("https://files/two.mp3")
    assert first != second
    assert Path(first).read_bytes() == b"https://files/one.mp3"
    assert Path(second).read_bytes() == b"https://files/two.mp3"


def test_invalid_request_has_an_explicit_stage_and_error_type(handler):
    module, _, _ = handler
    result = module.handler({"id": "bad-job", "input": {}})
    assert result["error_type"] == "INVALID_INPUT"
    assert result["stage"] == "validation"
    assert result["job_id"] == "bad-job"


@pytest.mark.parametrize("wrap", [False, True])
def test_exception_without_message_still_returns_a_nonempty_original_cause(handler, wrap):
    module, pipeline, _ = handler

    def fail(*args, **kwargs):
        try:
            raise EOFError()
        except EOFError as error:
            if wrap:
                raise Exception(str(error))
            raise

    pipeline.song_cover_pipeline = fail
    result = module.handler({"id": "empty-error", "input": {"song_input_url": "https://files/track.mp3", "rvc_model": "Singer"}})
    assert result["error"] == "EOFError"
    assert "EOFError" in result["traceback"]


def test_empty_download_is_rejected_before_voice_conversion(handler):
    module, pipeline, folder = handler
    (folder / "input.mp3").write_bytes(b"")
    calls = []
    pipeline.song_cover_pipeline = lambda *a, **kw: calls.append(a)
    result = module.handler({"id": "empty-file", "input": {"song_input_url": "https://files/track.mp3", "rvc_model": "Singer"}})
    assert result["error"] == "Downloaded audio file is empty"
    assert result["stage"] == "audio_download"
    assert calls == []
    assert not (folder / "input.mp3").exists()
