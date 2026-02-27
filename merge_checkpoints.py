"""
Checkpoint Averaging - усреднение весов последних N чекпоинтов.
"""

import os
import glob
import argparse
from collections import OrderedDict

import torch
import safetensors.torch
from transformers import AutoTokenizer

# ======Settings=========
MODELS_DIR = "models"
OUTPUT_DIR = "merged_model"
NUM_CHECKPOINTS = 7  # Сколько последних чекпоинтов усреднять
# ======Settings=========


def get_checkpoints(models_dir: str, n: int) -> list:
    """Получает последние n чекпоинтов."""
    checkpoints = glob.glob(os.path.join(models_dir, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"Не найдено чекпоинтов в {models_dir}")
    
    # Сортируем по номеру
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    
    # Берём последние n
    selected = checkpoints[-n:]
    return selected


def load_state_dict(checkpoint_path: str) -> dict:
    """Загружает state_dict из чекпоинта."""
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(weights_path):
        return safetensors.torch.load_file(weights_path)
    
    weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(weights_path):
        return torch.load(weights_path, map_location="cpu")
    
    raise FileNotFoundError(f"Не найдены веса в {checkpoint_path}")


def average_checkpoints(checkpoint_paths: list) -> OrderedDict:
    """Усредняет веса из нескольких чекпоинтов."""
    print(f"\nУсреднение {len(checkpoint_paths)} чекпоинтов:")
    for cp in checkpoint_paths:
        print(f"  - {os.path.basename(cp)}")
    
    # Загружаем первый чекпоинт как базу
    avg_state_dict = load_state_dict(checkpoint_paths[0])
    
    # Конвертируем в float для точного усреднения
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key].float()
    
    # Добавляем остальные чекпоинты
    for cp_path in checkpoint_paths[1:]:
        state_dict = load_state_dict(cp_path)
        for key in avg_state_dict:
            if key in state_dict:
                avg_state_dict[key] += state_dict[key].float()
    
    # Делим на количество чекпоинтов
    n = len(checkpoint_paths)
    for key in avg_state_dict:
        avg_state_dict[key] /= n
    
    # Возвращаем в оригинальный dtype (обычно float16/bfloat16)
    # Для safetensors лучше оставить float32, при загрузке модель сама сконвертит
    
    return OrderedDict(avg_state_dict)


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Averaging")
    parser.add_argument("-n", "--num", type=int, default=NUM_CHECKPOINTS,
                        help=f"Количество последних чекпоинтов для усреднения (default: {NUM_CHECKPOINTS})")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_DIR,
                        help=f"Папка для сохранения (default: {OUTPUT_DIR})")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHECKPOINT AVERAGING")
    print("=" * 60)
    
    # Получаем чекпоинты
    checkpoints = get_checkpoints(MODELS_DIR, args.num)
    print(f"\nНайдено чекпоинтов: {len(checkpoints)}")
    
    if len(checkpoints) < args.num:
        print(f"Внимание: запрошено {args.num}, но доступно только {len(checkpoints)}")
    
    # Усредняем
    avg_state_dict = average_checkpoints(checkpoints)
    
    # Создаём output директорию
    os.makedirs(args.output, exist_ok=True)
    
    # Копируем конфиг и токенайзер из последнего чекпоинта
    last_checkpoint = checkpoints[-1]
    
    # Сохраняем токенайзер
    tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
    tokenizer.save_pretrained(args.output)
    print(f"\nТокенайзер сохранён в {args.output}")
    
    # Копируем config.json
    import shutil
    config_src = os.path.join(last_checkpoint, "config.json")
    config_dst = os.path.join(args.output, "config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, config_dst)
        print(f"Config скопирован")
    
    # Сохраняем усреднённые веса
    weights_path = os.path.join(args.output, "model.safetensors")
    safetensors.torch.save_file(avg_state_dict, weights_path)
    print(f"Веса сохранены в {weights_path}")
    
    # Размер файла
    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"Размер: {size_mb:.1f} MB")
    
    print("\n" + "=" * 60)
    print(f"Готово! Модель сохранена в: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
