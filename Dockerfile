FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    SHELL=/bin/bash \
    HF_HOME=.cache/huggingface \
    TORCH_HOME=.cache/torch

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip python3-venv \
      git git-lfs wget curl ca-certificates \
      libglib2.0-0 libsm6 libgl1-mesa-glx libxrender1 libxext6 libsndfile1 \
      ffmpeg sox build-essential google-perftools \
      fonts-dejavu-core procps jq && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# ----- Torch stack (CUDA 11.8) -----
RUN python3 -m pip install --upgrade pip==24.0 setuptools wheel && \
    python3 -m pip install \
      --no-cache-dir \
      --timeout=120 \
      --retries=5 \
      torch==2.0.1+cu118 \
      torchvision==0.15.2+cu118 \
      torchaudio==2.0.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118

# ----- Project deps (как в cog.yaml), без requirements.txt -----
# Примечание: при желании можно зафиксировать onnxruntime-gpu версией, например:
# onnxruntime-gpu==1.16.3
RUN python3 -m pip install --no-cache-dir \
    deemix \
    fairseq==0.12.2 \
    faiss-cpu==1.7.3 \
    "ffmpeg-python>=0.2.0" \
    gradio==3.39.0 \
    lib==4.0.0 \
    librosa==0.9.1 \
    numpy==1.23.5 \
    onnxruntime-gpu \
    "praat-parselmouth>=0.4.2" \
    pedalboard==0.7.7 \
    pydub==0.25.1 \
    pyworld==0.3.4 \
    Requests==2.31.0 \
    scipy==1.11.1 \
    soundfile==0.12.1 \
    torchcrepe==0.0.20 \
    tqdm==4.65.0 \
    yt_dlp==2023.7.6 \
    sox==1.4.1 \
    "imageio[ffmpeg]" \
    "gradio"


# ----- Gradio + websockets fix -----
RUN python3 -m pip install --no-cache-dir --upgrade \
    "websockets>=13.0" \
    "gradio>=4.0.0"

# ----- RunPod SDK -----
RUN python3 -m pip install --no-cache-dir runpod

COPY . .

# Prepare dirs
RUN mkdir -p rvc_models mdxnet_models song_output .cache/huggingface .cache/torch

# Entry
COPY --chmod=755 start_standalone.sh /start.sh
CMD ["/start.sh"]