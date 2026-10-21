"""Run inside the candidate image on real GPU hardware before serving jobs.

Exercise the failing reflect-pad/STFT kernel, cuDNN convolution, and the actual
vocal-separation ONNX model. Refuse a silent ONNX CPU fallback.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdx import MDX, MDXModel


def verify():
    if not torch.cuda.is_available():
        raise RuntimeError("Cover GPU preflight requires an NVIDIA CUDA device")
    device = torch.device("cuda:0")
    samples = torch.randn(1, 2, 8192, device=device)
    torch.nn.functional.pad(samples, (1024, 1024), mode="reflect")
    spectrum = torch.stft(samples[0], n_fft=2048, hop_length=512,
                          window=torch.hann_window(2048, device=device),
                          return_complex=True)
    if not torch.isfinite(spectrum).all().item():
        raise RuntimeError("CUDA STFT produced non-finite values")
    conv = torch.nn.Conv1d(2, 4, 3).to(device)
    if not torch.isfinite(conv(samples)).all().item():
        raise RuntimeError("cuDNN convolution produced non-finite values")

    root = Path(__file__).resolve().parents[1]
    model_path = root / "mdxnet_models" / "UVR-MDX-NET-Voc_FT.onnx"
    params_path = root / "mdxnet_models" / "model_data.json"
    metadata = json.loads(params_path.read_text())
    params = metadata[MDX.get_hash(str(model_path))]
    model = MDXModel(device, params["mdx_dim_f_set"],
                     2 ** params["mdx_dim_t_set"], params["mdx_n_fft_scale_set"])
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = "/tmp/cover_gpu_preflight"
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CUDAExecutionProvider"])
    providers = session.get_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("MDX ONNX session fell back to CPU: " + repr(providers))
    result = session.run(None, {"input": np.zeros((1, 4, model.dim_f, model.dim_t), dtype=np.float32)})
    if not all(np.isfinite(item).all() for item in result):
        raise RuntimeError("MDX ONNX inference produced non-finite values")
    profile_path = Path(session.end_profiling())
    try:
        profile = json.loads(profile_path.read_text())
        cuda_nodes = sum(event.get("args", {}).get("provider") == "CUDAExecutionProvider"
                         for event in profile)
    finally:
        profile_path.unlink(missing_ok=True)
    if not cuda_nodes:
        raise RuntimeError("MDX ONNX profiling recorded no CUDA compute nodes")
    torch.cuda.synchronize()
    report = {"gpu": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability(0)),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "onnxruntime": ort.__version__,
              "onnx_providers": providers, "onnx_cuda_nodes": cuda_nodes, "preflight": "passed"}
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    verify()
