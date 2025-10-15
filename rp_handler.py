import base64
import io
import os
import random
import tempfile
import time
from typing import Any, Dict, Optional

import torch
import torchaudio

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# Импорты из DiffRhythm
from infer.infer import inference
from infer.infer_utils import (
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)

LOGGER = RunPodLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SR = 44100

# Кэшим модели по длине (2048 | 6144)
_CACHE = {}

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _download_to_local(url: str) -> Optional[str]:
    """Скачивает файл по URL и возвращает локальный путь."""
    try:
        info = rp_file(url)
        return info["file_path"]
    except Exception as e:
        LOGGER.error(f"Download failed: {e}")
        return None


def _is_valid_duration(x: int) -> bool:
    """Проверяет валидность длительности."""
    return x == 95 or (96 <= x <= 285)


def _wav_b64_from_tensor(t: torch.Tensor, sr: int = SR) -> str:
    """Конвертирует тензор аудио в base64."""
    buf = io.BytesIO()
    torchaudio.save(buf, t.cpu(), sample_rate=sr, format="wav")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _load_models(max_frames: int, device: str):
    """Загружает и кэширует модели для заданного max_frames."""
    if max_frames not in _CACHE:
        LOGGER.info(f"Loading models for max_frames={max_frames} on {device}")
        cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
        _CACHE[max_frames] = {
            "cfm": cfm,
            "tokenizer": tokenizer,
            "muq": muq,
            "vae": vae
        }
    m = _CACHE[max_frames]
    return m["cfm"], m["tokenizer"], m["muq"], m["vae"]


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Handler для DiffRhythm inference.
    Input параметры:
      lyric: str - текст песни в LRC формате (может быть пустым)
      audio_length: int - длина аудио (95 или 96-285)
      music_duration: int - длительность музыки (по умолчанию = audio_length)
      ref_prompt: str - текстовое описание стиля
      ref_audio_url: str - URL референсного аудио для стиля
      chunked: bool - использовать chunked декодирование (по умолчанию True)
      seed: int - random seed (опционально)
    Должен быть указан ЛИБО ref_prompt ЛИБО ref_audio_url (не оба сразу).
    """
    t0 = time.time()
    inp = job.get("input", {}) or {}

    # Парсинг входных параметров
    lyric = inp.get("lyric", "")
    audio_length = int(inp.get("audio_length", 95))
    music_duration = int(inp.get("music_duration", audio_length))

    # Валидация длительностей
    if not _is_valid_duration(audio_length):
        return {"error": "audio_length must be 95 or between 96 and 285."}
    if not _is_valid_duration(music_duration):
        return {"error": "music_duration must be 95 or between 96 and 285."}

    # Определяем max_frames на основе audio_length
    max_frames = 2048 if audio_length == 95 else 6144

    # Валидация референса стиля
    ref_prompt = inp.get("ref_prompt", "")
    ref_audio_url = inp.get("ref_audio_url", "")

    if not (ref_prompt or ref_audio_url):
        return {"error": "Either 'ref_prompt' or 'ref_audio_url' must be provided."}  # noqa
    if ref_prompt and ref_audio_url:
        return {"error": "Use only one: 'ref_prompt' OR 'ref_audio_url'."}

    chunked = bool(inp.get("chunked", True))
    seed = int(inp.get("seed", random.randint(0, 2**31 - 1)))
    torch.manual_seed(seed)

    try:
        # Загружаем модели
        cfm, tokenizer, muq, vae = _load_models(max_frames, DEVICE)

        LOGGER.info(f"Processing: audio_length={audio_length}, chunked={chunked}, seed={seed}")  # noqa

        # Получаем токены лирики
        lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
            max_frames, lyric, tokenizer, music_duration, DEVICE
        )

        # Получаем style prompt
        if ref_prompt:
            # Текстовый промпт
            LOGGER.info(f"Using text prompt: {ref_prompt}")
            style_prompt = get_style_prompt(muq, prompt=ref_prompt)
        else:
            # Аудио референс
            LOGGER.info(f"Downloading audio reference: {ref_audio_url}")
            local_audio_path = _download_to_local(ref_audio_url)
            if not local_audio_path:
                return {"error": "Failed to download 'ref_audio_url'."}

            LOGGER.info(f"Using audio reference: {local_audio_path}")
            style_prompt = get_style_prompt(muq, wav_path=local_audio_path)

        # Получаем negative style prompt
        negative_style_prompt = get_negative_style_prompt(DEVICE)

        # Получаем латентный промпт и pred_frames
        latent_prompt, pred_frames = get_reference_latent(
            DEVICE,
            max_frames,
            False,  # edit
            None,   # edit_start_time
            None,   # edit_audio_path
            vae
        )

        # Inference с autocast
        LOGGER.info("Starting inference...")
        with torch.autocast(device_type="cuda",
                            enabled=(DEVICE == "cuda"),
                            dtype=DTYPE):
            out = inference(
                cfm_model=cfm,
                vae_model=vae,
                cond=latent_prompt,
                text=lrc_prompt,
                duration=end_frame,
                style_prompt=style_prompt,
                negative_style_prompt=negative_style_prompt,
                start_time=start_time,
                pred_frames=pred_frames,
                batch_infer_num=1,
                song_duration=song_duration,
                chunked=chunked
            )

        # Проверяем результат
        if not isinstance(out, list) or len(out) == 0:
            return {"error": "Inference returned empty output."}

        # Обрабатываем результат
        audio_i16 = out[0]

        # Приводим к правильной форме (channels, samples)
        if audio_i16.ndim == 1:
            audio_i16 = audio_i16[None, :]
        elif audio_i16.ndim == 2 and audio_i16.shape[0] > audio_i16.shape[1]:
            audio_i16 = audio_i16.T

        # Сохраняем во временный файл
        tmp = tempfile.mkdtemp()
        out_wav = os.path.join(tmp, "output.wav")
        torchaudio.save(out_wav, audio_i16.cpu(), sample_rate=SR)

        LOGGER.info(f"Audio saved to: {out_wav}")

        # Конвертируем в base64 для возврата
        b64 = _wav_b64_from_tensor(audio_i16, SR)

        elapsed = round(time.time() - t0, 3)
        LOGGER.info(f"Inference completed in {elapsed}s")

        return {
            "output_path": out_wav,
            "audio_base64": b64,
            "sample_rate": SR,
            "audio_length": audio_length,
            "seed": seed,
            "chunked": chunked,
            "time_sec": elapsed,
        }

    except torch.cuda.OutOfMemoryError as e:
        msg = str(e)
        LOGGER.error(f"CUDA OOM: {msg}")
        return {
            "error": "CUDA out of memory. Try audio_length=95 with chunked=True.",  # noqa
            "detail": msg
        }
    except RuntimeError as e:
        msg = str(e)
        if "CUDA out of memory" in msg:
            LOGGER.error(f"CUDA OOM: {msg}")
            return {
                "error": "CUDA out of memory. Try audio_length=95 with chunked=True.",  # noqa
                "detail": msg
            }
        LOGGER.error(f"Runtime error: {msg}")
        return {"error": msg}
    except Exception as e:
        import traceback
        trace = traceback.format_exc(limit=8)
        LOGGER.error(f"Error: {str(e)}\n{trace}")
        return {"error": str(e), "trace": trace}


if __name__ == "__main__":
    LOGGER.info("Starting RunPod Serverless Handler...")
    runpod.serverless.start({"handler": handler})
