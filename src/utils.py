"""
Общие утилиты для SFT Ablation Study.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from datetime import datetime

import yaml
import torch


def setup_logging(name: str = "sft_ablation", level=logging.INFO) -> logging.Logger:
    """Настройка логирования."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: str, base_config_path: str = None) -> dict:
    """
    Загрузка YAML конфига с наследованием от base.
    
    Args:
        config_path: Путь к конфигу эксперимента
        base_config_path: Путь к базовому конфигу (если None — ищем configs/base.yaml)
    
    Returns:
        Мёрженный конфиг
    """
    # Загружаем базовый конфиг
    if base_config_path is None:
        project_root = Path(__file__).parent.parent
        base_config_path = project_root / "configs" / "base.yaml"

    config = {}
    if Path(base_config_path).exists():
        with open(base_config_path) as f:
            config = yaml.safe_load(f) or {}

    # Мёржим с конфигом эксперимента
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            exp_config = yaml.safe_load(f) or {}
        config = deep_merge(config, exp_config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивный мёрж словарей."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def extract_boxed_answer(text: str) -> str | None:
    """
    Извлечение ответа из \\boxed{...}.
    
    Поддерживает вложенные скобки: \\boxed{\\frac{1}{2}}
    """
    # Ищем последний \\boxed{...}
    pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def save_results(results: dict, output_path: str):
    """Сохранение результатов в JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Добавляем метаданные
    results["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.getLogger("sft_ablation").info(f"Results saved to {output_path}")


def load_results(path: str) -> dict:
    """Загрузка результатов из JSON."""
    with open(path) as f:
        return json.load(f)


def get_gpu_memory_info() -> dict:
    """Информация о GPU памяти."""
    if not torch.cuda.is_available():
        return {"available": False}

    info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        info[f"gpu_{i}"] = {
            "name": props.name,
            "total_gb": round(props.total_mem / 1e9, 2),
            "allocated_gb": round(allocated / 1e9, 2),
            "reserved_gb": round(reserved / 1e9, 2),
            "free_gb": round((props.total_mem - allocated) / 1e9, 2),
        }
    return info


class Timer:
    """Простой контекстный менеджер для замера времени."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            logging.getLogger("sft_ablation").info(
                f"[{self.name}] Elapsed: {self.elapsed:.1f}s"
            )
