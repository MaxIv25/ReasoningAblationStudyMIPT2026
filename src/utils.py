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
    
    Использует stack-based matching для корректной обработки
    произвольной вложенности скобок:
        \\boxed{\\frac{1}{\\sqrt{2}}}  — 3 уровня
        \\boxed{\\left(\\frac{a}{b}\\right)}  — 2 уровня
    
    Берёт ПОСЛЕДНИЙ \\boxed{...} в тексте (финальный ответ).
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1].strip()
    return None


def normalize_latex(s: str) -> str:
    """
    Normalize LaTeX string for robust comparison.

    Handles the most common formatting variants that are mathematically
    equivalent but differ in LaTeX representation:
      - \\dfrac, \\tfrac → \\frac
      - \\left( / \\right) → ( / )
      - \\displaystyle → remove
      - \\text{...} → contents only (for \\text{Evelyn} vs Evelyn)
      - Whitespace normalization
    """
    # Display-style fraction variants → standard \frac
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")

    # Remove \displaystyle
    s = s.replace("\\displaystyle", "")

    # \left( → (  ,  \right) → )  etc.
    s = re.sub(r"\\left\s*([(\[{|.])", r"\1", s)
    s = re.sub(r"\\right\s*([)\]}|.])", r"\1", s)

    # \text{...} → contents
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    # \textbf, \textit, \mathrm, \mathbf, etc.
    s = re.sub(
        r"\\(?:textbf|textit|mathrm|mathbf|mathit|operatorname)\s*\{([^}]*)\}",
        r"\1",
        s,
    )

    # Strip surrounding $ signs
    s = s.strip().strip("$").strip()

    # Normalize whitespace: collapse multiple spaces to one
    s = re.sub(r"\s+", " ", s).strip()

    return s


def verify_answer(predicted: str | None, ground_truth: str) -> bool:
    """
    Verify if predicted answer matches ground truth.

    Pipeline:
      1. Normalize LaTeX (dfrac→frac, strip \\left/\\right, etc.)
      2. Exact string match after normalization
      3. Try math_verify on normalized strings
      4. Try sympy-based symbolic comparison for numeric expressions
      5. Fall back to normalized string comparison (strip all formatting)
    """
    if predicted is None:
        return False

    predicted = normalize_latex(predicted)
    ground_truth = normalize_latex(ground_truth)

    # Exact match after normalization (catches most dfrac/frac cases)
    if predicted == ground_truth:
        return True

    # Try math_verify
    try:
        from math_verify import parse, verify
        result = verify(parse(ground_truth), parse(predicted))
        if result:
            return True
    except Exception:
        pass

    # Try sympy for numeric/algebraic expressions
    try:
        import sympy
        gt_expr = sympy.sympify(ground_truth.replace("\\", ""))
        pred_expr = sympy.sympify(predicted.replace("\\", ""))
        if sympy.simplify(gt_expr - pred_expr) == 0:
            return True
    except Exception:
        pass

    # Fallback: strip all formatting and compare
    def _strip_compare(s):
        s = s.replace("\\$", "").replace("$", "")
        s = s.replace("\\,", "").replace(",", "")
        s = s.replace(" ", "").strip()
        try:
            return float(s)
        except ValueError:
            return s.lower()

    return _strip_compare(predicted) == _strip_compare(ground_truth)


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
            "total_gb": round(props.total_memory / 1e9, 2),
            "allocated_gb": round(allocated / 1e9, 2),
            "reserved_gb": round(reserved / 1e9, 2),
            "free_gb": round((props.total_memory - allocated) / 1e9, 2),
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
