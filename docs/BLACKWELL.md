# CUDA 12.8 cover worker

The October 21 failures on RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
occurred in torch reflect padding/STFT with the legacy torch 2.0.1+cu118 image.
The same worker handled both reported failures. Successful jobs on Ampere/Ada
were consistent with an architecture mismatch.

## Build

Build from a clean committed checkout, keeping the immutable base digest in
`Dockerfile.blackwell`:

```sh
docker build -f Dockerfile.blackwell -t sintecs/cover_music:blackwell-20261021-candidate2 .
docker push sintecs/cover_music:blackwell-20261021-candidate2
```

Pinned inference stack: Python 3.10 from the tested base; torch 2.7.1+cu128,
torchaudio 2.7.1+cu128, torchvision 0.22.1+cu128, onnxruntime-gpu 1.22.0,
numpy 1.26.4. PyTorch wheels provide CUDA 12.8/cuDNN 9 libraries. The inherited
CUDA 11.8 development toolkit is unused by inference and is not upgraded.
The container requires a host driver supporting CUDA 12.8 or newer.
The old base and its baked voice models remain in lower image layers; allow
additional image storage and cold-start transfer time for the new runtime.

`start_blackwell.sh` runs `scripts/verify_gpu_runtime.py` before the existing
handler. It verifies CUDA reflect padding/STFT, cuDNN convolution, and an
actual inference through the baked vocal-separation ONNX model. ONNX profiling
must record CUDA compute nodes; a silent CPU fallback fails startup.

HuBERT uses explicit trusted checkpoint loading only at the baked path, followed
by fairseq's configuration/state upgrade. Custom RVC checkpoints keep restricted
weights-only loading. No global environment override disables that restriction.
Custom archive directory names must remain a single component beneath rvc_models.
Pitch quantization uses np.int64, compatible with NumPy 1.26.
The HTTP/RunPod input and output contract is unchanged.

## Verification and rollout

Hardware acceptance is pending. Do not deploy this candidate on that basis.
The live endpoint remains on `fix-20261019-88fbaa7` with Blackwell MIG excluded.

For acceptance, use separate endpoints with max one worker and force exact GPU
types through pool exclusions. Run a full cover with built-in and custom voices,
short mono/stereo MP3/WAV/M4A inputs, long input, invalid input followed by a fresh
successful request on the same worker. Fully decode outputs and check finite,
non-silent samples and preserved duration. Record actual worker GPU types and
image digest, then stop test workers.

Once accepted, deployment requires replacing the RunPod image and removing the
Blackwell type exclusion together. Rebuilding backend Compose services alone
does not replace this external GPU worker image. Rollback restores the old image
and the Blackwell MIG exclusion; keep the previous tag and digest available.

References: [PyTorch 2.7 Blackwell/CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/),
[official wheel version combinations](https://pytorch.org/get-started/previous-versions/),
[ONNX CUDA/cuDNN compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[PyTorch checkpoint loading](https://docs.pytorch.org/docs/stable/notes/serialization).
