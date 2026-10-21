# CUDA 12.8 cover worker

The October 21 failures on RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
occurred in torch reflect padding/STFT with the legacy torch 2.0.1+cu118 image.
The same worker handled both reported failures. Successful jobs on Ampere/Ada
were consistent with an architecture mismatch.

## Build

Build from a clean committed checkout, keeping the immutable base digest in
`Dockerfile.blackwell`:

```sh
docker build -f Dockerfile.blackwell -t sintecs/cover_music:blackwell-20261021-c920723 .
docker push sintecs/cover_music:blackwell-20261021-c920723
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

Published image: **`sintecs/cover_music:blackwell-20261021-c920723`**.
Immutable digest:

```text
sintecs/cover_music@sha256:3d138b9a9e069285eb41229f8376b2e2c180a226831e1f969bc053845e917767
```

Source implementation commit: `c920723`. Complete Linux worker suite: **32 passed**;
`pip check`: no broken requirements. CPU startup correctly rejects absent CUDA.

Measured hardware acceptance on 21 October 2026:

| GPU | Successful audio results | Expected empty-input failures | Worker |
|---|---:|---:|---|
| RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb | 8 | 1 | `db7tqny7a8svmd` |
| RTX 4090 | 7 | 1 | `ov1o9dbijt3trt` |
| RTX A4000 (16 GB) | 3 | 0 | `m4g9wiajtvt2j4` |

The tested workers completed full RVC covers using SonGoku and custom WillyWill4Real
voices. Inputs included MP3, WAV, M4A, mono and stereo; output contracts MP3 and
WAV both passed. A 358.217143-second MP3 returned decoded audio of 358.14 seconds
on Blackwell and 358.12 seconds on RTX 4090 (container/encoder padding included
in input duration). Default full long-cover execution was 52.590 and 43.952 seconds,
respectively. The additional Blackwell custom-voice long cover reused the
pipeline's cached separation and took 14.704 seconds; it is not a fresh separation
benchmark. Built-in/custom output signals differ.

All 18 results fully decode, contain finite nonzero samples, and preserve input
duration within 0.12 seconds. After each empty-input failure, the same worker
processed an input with a previously unseen file hash, exercising fresh inference.
RTX A4000 (16 GB) also passed built-in/custom short covers and the full long
cover in 58.277 seconds, with 358.14 seconds of decoded output.

First-job queue delay with Flashboot OFF was 263.419 seconds on RTX 4090 and
362.522 seconds on Blackwell, including placement, image transfer and startup.
The first RTX A4000 job queued for 312.331 seconds.
The uncompressed image is 34,841,033,603 bytes (32.45 GiB). One discarded RTX 4090
provisioning attempt hit Docker Hub's anonymous pull quota; a subsequent worker
successfully started. No registry credentials changed during acceptance.

All three acceptance endpoints were scaled to minimum/maximum zero after their
jobs finished, and the temporary audio server was stopped (port verified closed).

Production rollout completed on 22 October 2026 (Asia/Yekaterinburg;
21 October UTC). Existing endpoint `ys5oqibu9wy1y4` is version **14**, using
template `9na0iw11jc` and the immutable image digest above. Read-back through
GraphQL and REST v2 confirmed minimum CUDA **12.8**, no exact-version whitelist,
and pools `AMPERE_16,ADA_24,AMPERE_24` with the Blackwell MIG exclusion removed.
Environment values, disk 25 GB, workers 0–5, idle timeout 5 seconds, queue-delay
scaling 4, Flashboot and execution timeout 700000 ms were preserved.

Production smoke job `84b2294d-3aad-4d68-a16c-7eea6a969ed2-e2` completed on
new-image RTX 4090 worker `9nbva65aswkhbb`: 7.128 seconds execution, 21.236 seconds
queue delay. Startup reported torch 2.7.1+cu128, CUDA 12.8 and **178 ONNX CUDA
compute nodes**. Its stereo MP3 fully decodes to 6.72 seconds from a 6.765714-second
input, with finite non-silent samples (RMS 0.04863). An earlier smoke job completed
on an outdated worker during rolling replacement and was not counted as new-image
validation. These direct provider checks bypassed Redis and notifications.
The temporary fixture server was stopped again after validation.

The original template `thwz2gfxhh` remains available, plus rollback copy
`r6214sfrk0` using `fix-20261019-88fbaa7`. Restore the old image, Blackwell MIG
exclusion and previous empty CUDA floor together if rolling back.

For acceptance, use separate endpoints with max one worker and force exact GPU
types through pool exclusions. Run a full cover with built-in and custom voices,
short mono/stereo MP3/WAV/M4A inputs, long input, invalid input followed by a fresh
successful request on the same worker. Fully decode outputs and check finite,
non-silent samples and preserved duration. Record actual worker GPU types and
image digest, then stop test workers.

For subsequent deployments, replace the RunPod image, set
`gpu.minCudaVersion` to `"12.8"`, and removing the Blackwell type exclusion.
A newer host CUDA/driver version such as 13.x is also suitable; do not restrict
the allowed versions to exactly 12.8. Rebuilding backend Compose services alone
does not replace this external GPU worker image. Rollback restores the old image
and the Blackwell MIG exclusion; keep the previous tag and digest available.
Always read back the actual image: the GraphQL `saveEndpoint` update accepted
`templateId` but retained the old template in this rollout. REST v1 PATCH applied
the template replacement while REST v2 was temporarily unavailable. Prefer the
current REST v2 endpoint image update for future releases and verify its result.

References: [PyTorch 2.7 Blackwell/CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/),
[official wheel version combinations](https://pytorch.org/get-started/previous-versions/),
[ONNX CUDA/cuDNN compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[PyTorch checkpoint loading](https://docs.pytorch.org/docs/stable/notes/serialization).
