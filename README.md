# Reasoning Ablation Study — MIPT 2026

**SFT Ablation Study для обучения reasoning LLM**

Проект по курсу оптимизации: систематическое исследование влияния различных компонентов SFT-этапа на качество reasoning в LLM.

## Модель и данные

| Компонент | Описание |
|-----------|----------|
| **Ученик** | Qwen3.5-4B-Base (4B dense, ctx 262K) |
| **Учитель** | Qwen3.5-35B-A3B (35B total, 3B activated MoE) |
| **Данные** | OpenR1-Math-220K (трассы DeepSeek-R1) |
| **Бенчмарки** | GSM8K test (1.3K) + MATH-500 |
| **GPU** | 1× H200 140GB |

## Фазы эксперимента

| Фаза | Что исследуем | Эксперименты |
|------|--------------|-------------|
| **0** | Baseline | Eval raw Qwen3.5-4B-Base |
| **1** | Источник данных | R1 traces vs Qwen traces vs KL-constraint |
| **2** | Метод обучения | Full FT vs LoRA vs DoRA vs PiSSA |
| **3** | Curriculum и данные | Correct/incorrect, easy→hard, prompt masking |
| **4** | Оптимизатор | AdamW vs constant LR vs SGD vs AdaFactor |

## Структура

```
├── configs/          # YAML конфиги экспериментов
├── src/              # Общие утилиты
│   ├── evaluate.py   # Eval на GSM8K/MATH-500 (vLLM)
│   ├── data_utils.py # Загрузка и подготовка данных
│   ├── train_sft.py  # SFT training (TRL + PEFT)
│   └── generate_traces.py  # Генерация трасс учителем
├── sft/
│   ├── exp0_baseline/    # Phase 0
│   ├── exp1_data/        # Phase 1
│   ├── exp2_method/      # Phase 2
│   ├── exp3_curriculum/  # Phase 3
│   └── exp4_optimizer/   # Phase 4
├── analysis/         # Графики и сравнения
└── docs/             # Документация
```

## Quick Start

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Phase 0: Baseline
python sft/exp0_baseline/run_baseline.py --model Qwen/Qwen3.5-4B-Base

# 3. Подготовка данных
python src/data_utils.py --config configs/base.yaml --output data/openr1_20k

# 4. SFT Training (пример: LoRA)
python src/train_sft.py --config configs/exp2_2_lora.yaml --data-dir data/openr1_20k --method lora

# 5. Eval после обучения
python src/evaluate.py --model outputs/exp2_lora --output results/exp2_lora.json
```

## Метрики

| Метрика | Описание |
|---------|----------|
| **Accuracy (pass@1)** | Greedy decode → extract `\boxed{}` → math_verify |
| **Format compliance** | % ответов с корректным `\boxed{}` |
| **Avg response length** | Средняя длина ответа в словах |
| **Train/Eval loss** | Логируется через TensorBoard |

## Авторы

MIPT, 2026 — Курс по методам оптимизации