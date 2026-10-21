"""Real CPU pitch extraction also exercises the shared NumPy quantization path."""
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("parselmouth")
from vc_infer_pipeline import VC


@pytest.mark.parametrize("shift", [-12, 0, 12])
def test_pitch_quantization_with_supported_numpy(shift):
    config = SimpleNamespace(x_pad=1, x_query=6, x_center=38, x_max=41,
                             is_half=False, device="cpu")
    converter = VC(40000, config)
    time = np.arange(16000, dtype=np.float64) / 16000
    audio = 0.2 * np.sin(2 * np.pi * 440 * time)
    coarse, hz = converter.get_f0("sine.wav", audio, 100, shift, "pm", 3, 128)
    assert coarse.dtype == np.int64
    assert coarse.shape == hz.shape == (100,)
    assert coarse.min() >= 1 and coarse.max() <= 255
    voiced = hz[hz > 0]
    assert len(voiced) > 80
    assert np.median(voiced) == pytest.approx(440 * 2 ** (shift / 12), rel=0.01)
