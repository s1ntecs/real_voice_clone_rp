FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    TOKENIZERS_PARALLELISM=false

# ----- System deps -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3-pip python3.9-venv \
    git wget curl ca-certificates \
    libgl1-mesa-glx \
    ffmpeg \
    sox \
    libsndfile1 \
    build-essential \
    google-perftools \
 && rm -rf /var/lib/apt/lists/*

# Make python3 point to 3.9
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

WORKDIR /workspace

# ----- Torch stack (CUDA 11.8) -----
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2+cu118

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
    "imageio[ffmpeg]"

# ----- RunPod SDK -----
RUN python3 -m pip install --no-cache-dir runpod

# ----- Project files -----
# (Убираем COPY requirements.txt, т.к. не используем)
RUN mkdir -p /workspace/src \
    /workspace/rvc_models \
    /workspace/mdxnet_models \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch \
    /workspace/song_output

COPY src/ /workspace/src/
COPY mdxnet_models/ /workspace/mdxnet_models/
COPY rvc_models/ /workspace/rvc_models/
COPY rp_handler.py /workspace/
COPY start_standalone.sh /workspace/

# (опционально) предзагрузка моделей
# RUN python3 -m pip install --no-cache-dir huggingface_hub && \
#     python3 /workspace/download_models.py
# или
# RUN python3 -m pip install --no-cache-dir huggingface_hub && \
#     python3 /workspace/download_models.py --no-full

RUN chmod +x /workspace/start_standalone.sh

ENTRYPOINT ["/workspace/start_standalone.sh"]
