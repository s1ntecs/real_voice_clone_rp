"""Checkpoint migration tests; full runtime dependencies live in the worker image."""
from argparse import Namespace
from pathlib import Path
import pickle

import pytest
import torch

pytest.importorskip("fairseq")
import rvc


def test_custom_voice_does_not_allow_arbitrary_pickle_objects(tmp_path):
    path = tmp_path / "untrusted_voice.pth"
    torch.save({"config": Namespace(unexpected=True)}, path)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        rvc.get_vc("cpu", False, None, path)


def test_baked_hubert_loads_in_eval_mode_on_cpu():
    path = rvc.BASE_DIR / "rvc_models" / "hubert_base.pt"
    if not path.is_file():
        pytest.skip("requires the immutable image's baked HuBERT checkpoint")
    model = rvc.load_hubert("cpu", False, str(path))
    assert not model.training
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float32


def test_hubert_trusted_pickle_loader_rejects_other_files(tmp_path):
    path = tmp_path / "hubert_base.pt"
    torch.save({"cfg": Namespace(unexpected=True)}, path)
    with pytest.raises(ValueError, match="baked HuBERT"):
        rvc.load_hubert("cpu", False, str(path))
