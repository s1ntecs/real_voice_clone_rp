import os
import sys
import shutil
import zipfile
import urllib.request
import urllib.parse
import tempfile
import base64
from typing import Any, Dict, Optional
from argparse import Namespace

import torch
import onnxruntime as ort

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# Добавляем src в путь для импорта main
sys.path.insert(0, os.path.abspath("src"))
import main as m

LOGGER = RunPodLogger()

# Устанавливаем переменные окружения
os.environ.setdefault("HF_HOME", "/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/.cache/torch")


def log_gpu_info():
    """Логирует информацию о GPU для диагностики"""
    LOGGER.info("="*60)
    LOGGER.info("🔍 GPU DIAGNOSTIC INFO:")
    LOGGER.info(f"PyTorch version: {torch.__version__}")
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        LOGGER.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        LOGGER.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            LOGGER.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            LOGGER.info(f"  Memory: {mem:.2f} GB")

    providers = ort.get_available_providers()
    LOGGER.info(f"ONNX Runtime providers: {providers}")

    if 'CUDAExecutionProvider' in providers:
        LOGGER.info("✅ ONNX Runtime CUDA support: AVAILABLE")
    else:
        LOGGER.info("⚠️ ONNX Runtime CUDA support: NOT AVAILABLE")

    LOGGER.info("="*60)


def download_online_model(url: str, dir_name: str) -> None:
    """Скачивает и распаковывает модель RVC."""
    LOGGER.info(f"Downloading voice model: {dir_name}")

    zip_name = url.split("/")[-1]
    extraction_folder = os.path.join(m.rvc_models_dir, dir_name)

    if os.path.exists(extraction_folder):
        LOGGER.info(f"Model {dir_name} already exists, skipping download")
        return

    # Обработка pixeldrain URLs
    if "pixeldrain.com" in url:
        url = f"https://pixeldrain.com/api/file/{zip_name}"

    try:
        urllib.request.urlretrieve(url, zip_name)

        LOGGER.info("Extracting model...")
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.is_dir():
                    continue

                os.makedirs(extraction_folder, exist_ok=True)

                with zip_ref.open(member) as source, open(
                    os.path.join(extraction_folder,
                                 os.path.basename(member.filename)), "wb"
                ) as target:
                    shutil.copyfileobj(source, target)

        # Удаляем временный zip
        if os.path.exists(zip_name):
            os.remove(zip_name)

        LOGGER.info(f"Model {dir_name} successfully downloaded!")

    except Exception as e:
        LOGGER.error(f"Failed to download model: {e}")
        raise


def download_audio_from_url(url: str) -> str:
    """Скачивает аудио файл по URL и возвращает локальный путь."""
    try:
        info = rp_file(url)
        return info["file_path"]
    except Exception as e:
        LOGGER.error(f"Failed to download audio: {e}")
        # Fallback на прямую загрузку
        temp_path = os.path.join(tempfile.gettempdir(), "input_audio.mp3")
        urllib.request.urlretrieve(url, temp_path)
        return temp_path


def audio_to_base64(file_path: str) -> str:
    """Конвертирует аудио файл в base64."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Handler для RVC Voice Conversion.

    Input параметры:
      song_input_url: str - URL аудио файла для обработки
      rvc_model: str - название модели RVC (по умолчанию "Squidward")
      custom_rvc_model_download_url: str - URL для загрузки кастомной модели
      pitch_change: str - изменение тона ("no-change", "male-to-female",
        "female-to-male")
      index_rate: float - контроль акцента AI (0-1, по умолчанию 0.5)
      filter_radius: int - медианная фильтрация (0-7, по умолчанию 3)
      rms_mix_rate: float - контроль громкости (0-1, по умолчанию 0.25)
      pitch_detection_algorithm: str - алгоритм определения тона
                                        ("rmvpe" или "mangio-crepe")
      crepe_hop_length: int - hop length для crepe (по умолчанию 128)
      protect: float - защита согласных (0-0.5, по умолчанию 0.33)
      main_vocals_volume_change: float - изменение громкости осн вокала(дБ)
      backup_vocals_volume_change: float - изменение громкости бэк-вокала (дБ)
      instrumental_volume_change: float - изменение громкости инструментала(дБ)
      pitch_change_all: float - изменение тональности всего трека
      reverb_size: float - размер реверберации (0-1)
      reverb_wetness: float - уровень реверберации (0-1)
      reverb_dryness: float - уровень сухого сигнала (0-1)
      reverb_damping: float - затухание реверберации (0-1)
      output_format: str - формат вывода ("mp3" или "wav")
    """

    try:
        inp = job.get("input", {}) or {}

        # Получение параметров с дефолтными значениями
        song_input_url = inp.get("song_input_url")
        if not song_input_url:
            return {"error": "song_input_url is required"}

        rvc_model = inp.get("rvc_model", "Squidward")
        custom_rvc_model_download_url = inp.get(
            "custom_rvc_model_download_url")
        pitch_change = inp.get("pitch_change", "no-change")
        index_rate = float(inp.get("index_rate", 0.5))
        if not 0 <= index_rate <= 1:
            return {"error": "index_rate must be between 0 and 1"}

        filter_radius = int(inp.get("filter_radius", 3))
        if not 0 <= filter_radius <= 7:
            return {"error": "filter_radius must be between 0 and 7"}

        rms_mix_rate = float(inp.get("rms_mix_rate", 0.25))
        pitch_detection_algorithm = inp.get("pitch_detection_algorithm",
                                            "rmvpe")
        crepe_hop_length = int(inp.get("crepe_hop_length", 128))
        protect = float(inp.get("protect", 0.33))
        main_vocals_volume_change = float(inp.get("main_vocals_volume_change",
                                                  0))
        backup_vocals_volume_change = float(inp.get(
            "backup_vocals_volume_change", 0))
        instrumental_volume_change = float(inp.get(
            "instrumental_volume_change", 0))
        pitch_change_all = float(inp.get("pitch_change_all", 0))
        reverb_size = float(inp.get("reverb_size", 0.15))
        reverb_wetness = float(inp.get("reverb_wetness", 0.2))
        reverb_dryness = float(inp.get("reverb_dryness", 0.8))
        reverb_damping = float(inp.get("reverb_damping", 0.7))
        output_format = inp.get("output_format", "mp3")

        if output_format not in ["mp3", "wav"]:
            return {"error": "output_format must be 'mp3' or 'wav'"}

        LOGGER.info(f"Processing with model: {rvc_model}")

        # Загрузка кастомной модели если указана
        if custom_rvc_model_download_url:
            custom_model_name = urllib.parse.unquote(
                custom_rvc_model_download_url.split("/")[-1]
            )
            custom_model_name = os.path.splitext(custom_model_name)[0]

            download_online_model(
                url=custom_rvc_model_download_url,
                dir_name=custom_model_name
            )
            rvc_model = custom_model_name

        # Загрузка входного аудио
        LOGGER.info("Downloading input audio...")
        song_input_path = download_audio_from_url(song_input_url)

        # Конвертация pitch_change в числовое значение
        if pitch_change == "no-change":
            pitch_value = 0
        elif pitch_change == "male-to-female":
            pitch_value = 1
        else:  # female-to-male
            pitch_value = -1

        # Проверка существования модели
        model_path = os.path.join(m.rvc_models_dir, rvc_model)
        if not os.path.exists(model_path):
            return {"error": f"Model {rvc_model} not found at {model_path}"}

        # Запуск pipeline
        LOGGER.info("Starting voice conversion pipeline...")
        cover_path = m.song_cover_pipeline(
            song_input_path,
            rvc_model,
            pitch_value,
            keep_files=False,
            main_gain=main_vocals_volume_change,
            backup_gain=backup_vocals_volume_change,
            inst_gain=instrumental_volume_change,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            f0_method=pitch_detection_algorithm,
            crepe_hop_length=crepe_hop_length,
            protect=protect,
            pitch_change_all=pitch_change_all,
            reverb_rm_size=reverb_size,
            reverb_wet=reverb_wetness,
            reverb_dry=reverb_dryness,
            reverb_damping=reverb_damping,
            output_format=output_format
        )

        LOGGER.info(f"Cover generated at: {cover_path}")

        # Конвертируем результат в base64
        audio_base64 = audio_to_base64(cover_path)

        # Опционально загружаем на S3 или возвращаем base64
        result = {
            "output_path": cover_path,
            "audio_base64": audio_base64,
            "format": output_format,
            "model_used": rvc_model
        }

        # Очистка временных файлов
        if os.path.exists(song_input_path) and song_input_path != cover_path:
            os.remove(song_input_path)

        return result

    except Exception as e:
        LOGGER.error(f"Error in handler: {str(e)}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    LOGGER.info("Starting RunPod Serverless Handler for RVC...")
    log_gpu_info()
    runpod.serverless.start({"handler": handler})
