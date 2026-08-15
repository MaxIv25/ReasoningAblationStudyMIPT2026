"""
Общие утилиты для SFT Ablation Study.
"""

import json
import os
import re
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

import yaml
import torch


_LATEX_ATOM = r"(?:\\[A-Za-z]+|[A-Za-z0-9])"
_COMPACT_FRACTION = re.compile(
    rf"\\frac\s*(?!\{{)({_LATEX_ATOM})\s*(?!\{{)({_LATEX_ATOM})"
)
_COMPACT_SQRT = re.compile(rf"\\sqrt\s*(?!\[|\{{)({_LATEX_ATOM})")
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_NUMERIC_BASE_PATTERN = re.compile(r"(\d+)\s*_\s*(?:\{(\d+)\}|(\d+))")


def _prepare_math_text(text: str) -> str:
    """Apply only semantics-preserving normalization before Math-Verify."""
    prepared = _THOUSANDS_SEPARATOR.sub("", text.strip())
    prepared = prepared.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    prepared = _COMPACT_FRACTION.sub(r"\\frac{\1}{\2}", prepared)
    prepared = _COMPACT_SQRT.sub(r"\\sqrt{\1}", prepared)
    return re.sub(r"(?<!\\)\bsqrt\(([^()]*)\)", r"\\sqrt{\1}", prepared)


def _numeric_base_annotation(text: str) -> tuple[str, str] | None:
    match = _NUMERIC_BASE_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1), match.group(2) or match.group(3) or ""


def _has_structural_mismatch(reference: str, prediction: str) -> bool:
    reference_base = _numeric_base_annotation(reference)
    return (
        reference_base is not None
        and _numeric_base_annotation(prediction) != reference_base
    )


def _parse_math_answer(text: str):
    """Parse an already extracted answer with the strict reward profile."""
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        LatexNormalizationConfig,
        parse,
    )

    normalization = LatexNormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=False,
        nits=False,
        boxed="all",
        equations=False,
    )
    is_main_thread = threading.current_thread() is threading.main_thread()
    prepared = _prepare_math_text(text)
    if not (prepared.startswith("$") and prepared.endswith("$")):
        prepared = f"${prepared}$"
    return parse(
        prepared,
        extraction_config=(
            LatexExtractionConfig(
                try_extract_without_anchor=True,
                boxed_match_priority=0,
                normalization_config=normalization,
            ),
            ExprExtractionConfig(try_extract_without_anchor=True),
        ),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=5 if is_main_thread else None,
        raise_on_error=False,
    )


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

    # Мёржим с конфигом эксперимента. Optional ``inherits`` is resolved
    # relative to the experiment file and layered on top of the default base.
    if config_path and Path(config_path).exists():
        config_path = Path(config_path)
        with open(config_path) as f:
            exp_config = yaml.safe_load(f) or {}
        inherited = exp_config.pop("inherits", None)
        if inherited:
            inherited_path = Path(inherited)
            if not inherited_path.is_absolute():
                inherited_path = config_path.parent / inherited_path
            config = load_config(
                str(inherited_path), base_config_path=str(base_config_path)
            )
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
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
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
    """Conservatively compare extracted answers for online RL reward.

    The verifier intentionally fails closed: only strict Math-Verify equivalence
    is accepted, and presentation normalization is limited to thousands
    separators. Structural annotations such as numeric bases are checked before
    symbolic comparison. This avoids reward-hacking false positives from the old
    fallback that removed every comma and most formatting.
    """
    if predicted is None or _has_structural_mismatch(ground_truth, predicted):
        return False

    try:
        from math_verify import verify

        reference = _parse_math_answer(ground_truth)
        prediction = _parse_math_answer(predicted)
        if not reference or not prediction:
            return False
        return bool(
            verify(
                reference,
                prediction,
                float_rounding=9,
                strict=True,
                timeout_seconds=(
                    5
                    if threading.current_thread() is threading.main_thread()
                    else None
                ),
                raise_on_error=False,
            )
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return False

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
