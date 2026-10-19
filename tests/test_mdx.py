"""Exercise separation with real torch STFT and an identity ONNX boundary."""
import numpy as np
import pytest
import torch

import mdx


def session():
    separator = mdx.MDX.__new__(mdx.MDX)
    separator.device = torch.device("cpu")
    separator.model = mdx.MDXModel(
        separator.device, dim_f=17, dim_t=16, n_fft=32, hop=8
    )
    separator.process = lambda spec: spec.cpu().numpy()
    separator.prog = None
    return separator


def audio(length=2049):
    samples = np.arange(length, dtype=np.float32)
    return np.stack([np.sin(samples * 0.1), np.cos(samples * 0.17)])


def test_original_inference_exception_reaches_caller():
    separator = session()

    def fail(spec):
        raise RuntimeError("ONNX inference failed: original cause")

    separator.process = fail
    with pytest.raises(RuntimeError, match="original cause"):
        separator.process_wave(audio())


def test_only_one_audio_window_is_transferred_at_a_time(monkeypatch):
    transfers = []
    original_to = torch.Tensor.to

    def track_to(tensor, *args, **kwargs):
        if tensor.ndim == 3 and tensor.shape[1] == 2:
            transfers.append(tensor.numel())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", track_to)
    separator = session()
    source = audio()
    result = separator.process_wave(source)
    assert transfers
    assert max(transfers) <= 2 * separator.model.chunk_size
    np.testing.assert_allclose(result, source, atol=1e-5)


@pytest.mark.parametrize("length", [1, 87, 88, 89, 2049])
def test_overlap_padding_preserves_samples_and_channels(length):
    source = audio(length)
    result = session().process_wave(source)
    assert result.shape == (2, length)
    np.testing.assert_allclose(result, source, atol=1e-5)


@pytest.mark.parametrize("source", [
    np.empty((2, 0)), np.zeros((2, 512)), np.full((2, 512), np.nan),
])
def test_invalid_audio_fails_before_model_or_cuda_initialization(monkeypatch, tmp_path, source):
    monkeypatch.setattr(mdx.librosa, "load", lambda *a, **kw: (source, 44100))
    with pytest.raises(ValueError, match="audio|Audio"):
        mdx.run_mdx({}, str(tmp_path), "unused.onnx", "input.wav")
