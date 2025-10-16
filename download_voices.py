#!/usr/bin/env python3
"""
Скрипт для автоматического скачивания голосовых моделей RVC
из словаря VOICES в src/voices.py
"""

import os
import sys
import zipfile
import requests
import shutil
from pathlib import Path
from urllib.parse import urlparse

# Импортируем словарь VOICES из корневой директории
from voices import VOICES

# Константы
RVC_MODELS_DIR = Path(__file__).parent / 'rvc_models'
TEMP_DIR = Path(__file__).parent / 'temp_downloads'
CHUNK_SIZE = 8192  # Размер чанка для скачивания (8KB)


def download_file(url, destination):
    """
    Скачивает файл с progress bar
    
    Args:
        url: URL для скачивания
        destination: Путь для сохранения файла
    """
    print(f"  📥 Скачивание: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  ⏳ Прогресс: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print()  # Новая строка после завершения
        return True
        
    except Exception as e:
        print(f"\n  ❌ Ошибка при скачивании: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    Извлекает содержимое zip архива
    
    Args:
        zip_path: Путь к zip файлу
        extract_to: Куда извлечь
    """
    print(f"  📦 Распаковка архива...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"  ❌ Ошибка при распаковке: {e}")
        return False


def find_model_files(directory):
    """
    Ищет файлы моделей (.pth и .index) в директории и поддиректориях
    
    Args:
        directory: Директория для поиска
        
    Returns:
        Путь к директории с файлами модели или None
    """
    directory = Path(directory)
    
    # Сначала ищем в самой директории
    pth_files = list(directory.glob('*.pth'))
    if pth_files:
        return directory
    
    # Ищем в поддиректориях (первый уровень)
    for subdir in directory.iterdir():
        if subdir.is_dir():
            pth_files = list(subdir.glob('*.pth'))
            if pth_files:
                return subdir
    
    return None


def process_voice_model(voice_name, url):
    """
    Обрабатывает одну голосовую модель: скачивает, распаковывает, перемещает
    
    Args:
        voice_name: Название модели (ключ из словаря VOICES)
        url: URL для скачивания
        
    Returns:
        True если успешно, False если произошла ошибка
    """
    print(f"\n{'='*60}")
    print(f"🎤 Обработка модели: {voice_name}")
    print(f"{'='*60}")
    
    # Проверяем, существует ли уже модель
    target_dir = RVC_MODELS_DIR / voice_name
    if target_dir.exists():
        print(f"  ✅ Модель уже существует: {target_dir}")
        print(f"  ⏭️  Пропускаем...")
        return True
    
    # Создаем временную директорию
    TEMP_DIR.mkdir(exist_ok=True)
    temp_model_dir = TEMP_DIR / voice_name
    temp_model_dir.mkdir(exist_ok=True)
    
    # Определяем имя файла для скачивания
    zip_filename = f"{voice_name}.zip"
    zip_path = temp_model_dir / zip_filename
    
    try:
        # Скачиваем файл
        if not download_file(url, zip_path):
            return False
        
        # Распаковываем архив
        extract_dir = temp_model_dir / 'extracted'
        extract_dir.mkdir(exist_ok=True)
        
        if not extract_zip(zip_path, extract_dir):
            return False
        
        # Находим директорию с файлами модели
        model_files_dir = find_model_files(extract_dir)
        
        if not model_files_dir:
            print(f"  ❌ Не найдены файлы модели (.pth) в архиве")
            return False
        
        print(f"  📁 Найдены файлы модели в: {model_files_dir.relative_to(temp_model_dir)}")
        
        # Перемещаем файлы модели в целевую директорию
        print(f"  📂 Перемещение в: {target_dir}")
        shutil.move(str(model_files_dir), str(target_dir))
        
        print(f"  ✅ Модель успешно установлена!")
        
        # Показываем содержимое
        files = list(target_dir.glob('*'))
        print(f"  📄 Файлы модели:")
        for file in files:
            print(f"     - {file.name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Очищаем временные файлы
        try:
            if temp_model_dir.exists():
                shutil.rmtree(temp_model_dir)
        except Exception as e:
            print(f"  ⚠️  Не удалось удалить временные файлы: {e}")


def main():
    """
    Основная функция - скачивает все модели из словаря VOICES
    """
    print("\n" + "="*60)
    print("🎵 ЗАГРУЗКА ГОЛОСОВЫХ МОДЕЛЕЙ RVC")
    print("="*60)
    
    # Создаем директорию для моделей если её нет
    RVC_MODELS_DIR.mkdir(exist_ok=True)
    
    print(f"\n📁 Директория моделей: {RVC_MODELS_DIR}")
    print(f"📊 Всего моделей в словаре: {len(VOICES)}")
    
    # Подсчитываем сколько моделей уже установлено
    existing_models = [name for name in VOICES.keys() 
                      if (RVC_MODELS_DIR / name).exists()]
    
    print(f"✅ Уже установлено: {len(existing_models)}")
    print(f"📥 Нужно скачать: {len(VOICES) - len(existing_models)}")
    
    # Статистика
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Обрабатываем каждую модель
    for i, (voice_name, url) in enumerate(VOICES.items(), 1):
        print(f"\n[{i}/{len(VOICES)}]")
        
        # Проверяем наличие модели перед обработкой
        if (RVC_MODELS_DIR / voice_name).exists():
            skipped_count += 1
            print(f"⏭️  {voice_name}: Уже установлена")
            continue
        
        result = process_voice_model(voice_name, url)
        
        if result:
            success_count += 1
        else:
            failed_count += 1
    
    # Финальный отчет
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*60)
    print(f"✅ Успешно скачано: {success_count}")
    print(f"⏭️  Пропущено (уже были): {skipped_count}")
    print(f"❌ Ошибок: {failed_count}")
    print(f"📊 Всего моделей: {len(VOICES)}")
    print("="*60)
    
    # Очищаем временную директорию
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR)
            print("\n🧹 Временные файлы удалены")
        except Exception as e:
            print(f"\n⚠️  Не удалось удалить временную директорию: {e}")
    
    print("\n✨ Готово!\n")


if __name__ == "__main__":
    main()