FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    TOKENIZERS_PARALLELISM=false

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3.10-venv git wget \
    espeak-ng ffmpeg libsndfile1 sox \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace


COPY requirements.txt .

# Устанавливаем PyTorch 2.6.0 с CUDA 12.4
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Устанавливаем зависимости проекта
RUN python3 -m pip install -r /workspace/requirements.txt

# Устанавливаем RunPod SDK
RUN python3 -m pip install runpod

# Копируем ВСЕ файлы
COPY . /workspace/


# ==========================================
# ✅ ПРЕДЗАГРУЗКА МОДЕЛЕЙ (РЕКОМЕНДУЕТСЯ!)
# ==========================================
# Раскомментируй эти строки чтобы упаковать модели в образ
# Это увеличит размер образа до ~12GB но ускорит cold start до 10-20 секунд

# RUN python3 -m pip install huggingface_hub && \
#     python3 /workspace/download_models.py

# Альтернатива: загружать только base модель (экономия 3.5GB):
# RUN python3 -m pip install huggingface_hub && \
#     python3 /workspace/download_models.py --no-full

# ==========================================
# ЕСЛИ НЕ ХОЧЕШЬ ПРЕДЗАГРУЗКУ:
# ==========================================
# Закомментируй строки выше с RUN python3... download_models.py
# Модели загрузятся автоматически при первом запуске (займёт 3-5 минут)
# ==========================================

# Создаём кэш директории
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch

# Добавляем путь к Python
# ENV PYTHONPATH="${PYTHONPATH}:/workspace"
COPY --chmod=755 start_standalone.sh /start.sh
ENTRYPOINT ["/workspace/start_standalone.sh"]