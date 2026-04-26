# Reasoning Ablation Study — MIPT 2026

**Ablation study методов SFT и GRPO для обучения LLM рассуждению**

Проект исследует полный пайплайн обучения reasoning LLM (SFT → GRPO → RS+SFT → final GRPO), воспроизводя подход DeepSeek-R1 на модели Qwen3.5-0.8B-Base.

## Модель и данные

| Компонент | Описание |
|-----------|----------|
| **Модель** | Qwen3.5-0.8B-Base |
| **Данные** | OpenR1-Math-220K (20K filtered, DeepSeek-R1 traces) |
| **Бенчмарки** | GSM8K test (1.3K), MATH-500 |
| **Фреймворки** | TRL, PEFT, vLLM, math-verify |

## План экспериментов

### Этап 1: SFT Ablation

| # | Эксперимент | Гипотеза | Статус |
|---|-------------|----------|--------|
| 1 | Full FT vs LoRA vs DoRA vs PiSSA | PEFT неявно ограничивает KL к исходной модели за счёт низкого ранга. Full FT обновляет все веса → ниже loss, но больший дрифт от π₀ | 🔄 Full FT и LoRA запущены |
| 2 | Curriculum (random vs easy→hard) | Упорядочивание по сложности ≈ lr warm-up: начинаем с low-variance градиентов | ⏳ |
| 3 | Prompt masking vs loss на промпте | Loss на промпте может улучшить связь "условие → решение", но разбавит сигнал | ⏳ |

### Этап 2: GRPO Ablation

| # | Эксперимент | Суть |
|---|-------------|------|
| 4 | Vanilla GRPO | Baseline: REINFORCE + group baseline + IS + clipping + KL |
| 5 | Dr. GRPO | Убираем bias: Â = r - r̄ (без деления на σ) |
| 6 | DAPO | 4 трюка: asymmetric clip, dynamic sampling, token-level norm, overlong penalty |
| 7 | DPO-inspired baseline | Объединение аналитического результата DPO с итеративным подходом GRPO |

### Этап 3: Полный пайплайн

```
SFT (лучший метод) → GRPO (лучший вариант) → Rejection Sampling + SFT → Final GRPO
```

### Дополнительно (если позволит время)

- PRIME — token-level credit assignment
- Online/Offline DPO
- KL-constraint distillation (SFT + KL к учителю)

## Структура

```
├── configs/              # YAML конфиги экспериментов
│   ├── base.yaml         # Базовый конфиг (наследуется)
│   ├── exp1_*.yaml       # Фаза данных
│   ├── exp2_*.yaml       # Фаза методов (Full FT / LoRA / DoRA / PiSSA)
│   └── exp3_*.yaml       # Curriculum
├── src/
│   ├── train_sft.py      # SFT training (TRL + PEFT)
│   ├── evaluate.py       # Eval через vLLM (GSM8K, MATH-500)
│   ├── data_utils.py     # Загрузка, фильтрация, curriculum sorting
│   ├── generate_traces.py # Генерация traces учителем
│   └── utils.py          # Конфиги, логирование, extract_boxed
├── sft/                  # Скрипты запуска по фазам
├── logs/                 # Логи текущих экспериментов
├── analysis/             # Сравнение результатов
└── docs/                 # Документация
```

## Quick Start

```bash
# Установка
pip install -r requirements.txt

# Подготовка данных
python src/data_utils.py --config configs/base.yaml --output data/openr1_20k

# SFT Training (пример: LoRA)
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --config configs/exp2_2_lora.yaml \
    --data-dir data/openr1_20k \
    --method lora

# Eval
python src/evaluate.py --model outputs/exp2_lora --output results/exp2_lora.json
```

## Метрики

| Метрика | Описание |
|---------|----------|
| **Accuracy (pass@1)** | Sampling t=0.6, top_p=0.95 → extract `\boxed{}` → math_verify |
| **Format compliance** | % ответов с корректным `<think>...</think> \boxed{...}` |
| **Eval loss / token accuracy** | TensorBoard |
| **Compute efficiency** | Время обучения, пиковый VRAM |

## Текущий прогресс

- [x] Инфраструктура: SFT pipeline, eval pipeline, data pipeline
- [x] Конфиги для всех SFT экспериментов
- [/] Exp 1: Full FT — ~50%, eval_loss=0.477, token_acc=84.6%
- [/] Exp 2: LoRA r=64 — ~57%
- [ ] DoRA, PiSSA
- [ ] Curriculum, prompt masking
- [ ] GRPO реализация
- [ ] Полный пайплайн

## Авторы

MIPT, 2026 — Курс по методам оптимизации