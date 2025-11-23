import os
import zipfile
import tarfile
from pathlib import Path
from config import PATHS
import requests


def download_from_yandex_disk(public_url: str, save_path: str):
    """
    Скачивает файл с Яндекс.Диска по ссылке.
    """
    print("Начинаю загрузку с Яндекс.Диска...")
    print(f"Публичная ссылка: {public_url}")
    print(f"Файл будет сохранён в: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
       
        api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        params = {"public_key": public_url}

        print("Запрос прямой ссылки...")
        response = requests.get(api_url, params=params)
        response.raise_for_status()

        href = response.json().get("href")
        if not href:
            print("Не удалось получить прямую ссылку (href).")
            return False

        print("Прямая ссылка получена, начинаю скачивание...")

        
        file_response = requests.get(href, stream=True)
        file_response.raise_for_status()

        total_size = int(file_response.headers.get("content-length", 0))
        downloaded = 0

        with open(save_path, "wb") as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\rПрогресс: {percent:.1f}%", end="")

        print(f"\nФайл загружен: {save_path}")
        return True

    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str):
    """
    Разархивирует zip или tar архив.
    """
    print(f"Начинаю разархивирование {archive_path}")
    os.makedirs(extract_to, exist_ok=True)

    archive_path = Path(archive_path)

    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

        elif archive_path.suffix in ['.tar', '.gz', '.tgz'] or '.tar.' in archive_path.name:
            if archive_path.suffix in ['.gz', '.tgz']:
                mode = 'r:gz'
            elif archive_path.suffix == '.bz2':
                mode = 'r:bz2'
            else:
                mode = 'r'

            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print("Неизвестный формат архива.")
            return False

        print(f"Разархивировано в: {extract_to}")
        return True

    except Exception as e:
        print(f"Ошибка разархивирования: {e}")
        return False


if __name__ == "__main__":
    print("Запуск get_raw_data.py")

    url = "https://disk.yandex.ru/d/XPthmNk_pqEDaQ"

    print("PATHS:", PATHS) 

    archive_path = PATHS['raw_data']
    extract_path = PATHS['extracted_data']

    print(f"\nСкачивание данных с: {url}")
    print(f"Сохранение архива в: {archive_path}")

    ok = download_from_yandex_disk(url, archive_path)

    if ok:
        print("\nНачинаю разархивирование")
        extract_archive(archive_path, extract_path)
    else:
        print("Ошибка при скачивании файла")
