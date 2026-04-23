# SFT Ablation Study: Полный план

## Компоненты эксперимента

### Модель-ученик: ✅ Qwen3.5-4B-Base
- 4B dense, 32 layers, hidden_dim 2560, ctx 262K
- Архитектура: **Qwen3.5** (Gated DeltaNet + Gated Attention, FFN, **НЕ MoE — dense**)
- Layout: 8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
- Vocab: 248320, tied embeddings, **pre-trained only** (нет instruct)
- Control tokens `<|im_start|>` / `<|im_end|>` обучены → LoRA без дообучения embeddings
- HF: https://huggingface.co/Qwen/Qwen3.5-4B-Base

### Модель-учитель: ✅ Qwen3.5-35B-A3B
- 35B total, **3B activated** (MoE, 256 experts, 8+1 active)
- Архитектура: **Qwen3.5** (то же семейство, что и ученик!)
- Post-trained мультимодальная модель (vision+language)
- Контекст: 262K tokens
- HF: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

**✅ Ученик и учитель — одно семейство Qwen3.5 → одинаковые токенайзеры (vocab 248320).** Logit-based KL-дистилляция работает напрямую, без хаков.

**Плюс Q3.5-35B-A3B:** активируется всего 3B параметров → очень быстрый инференс на H200, сравнимый с 4B моделью. Генерация трасс будет быстрой.

### Бенчмарки: ✅ GSM8K + MATH-500
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k): ~8.5K задач (7.5K train, 1.3K test). Уровень: школьная математика.
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500): 500 задач из MATH benchmark. Уровень: олимпиадная математика.

**Рекомендация:** Оценивать на **обоих**. GSM8K — лёгкий (модель должна набирать >60%), MATH-500 — сложный (дифференцирует модели лучше). Если модель не набирает >50% на GSM8K после SFT — что-то сломалось.

### Трассы DeepSeek: ✅ OpenR1-Math-220K
- 94K задач в `default` split (лучший для SFT)
- 2-4 трассы на задачу от DeepSeek-R1-671B
- Проверены через Math-Verify
- Лимит 16K токенов на трассу

---

## Фазы эксперимента

### ФАЗА 0: Baseline (1-2 часа)

```
Exp 0.1: Qwen3.5-4B-Base → GSM8K test + MATH-500
         (zero-shot, без SFT)
         Промпт: "Please reason step by step, and put your final answer within \boxed{}."
         
Ожидание: GSM8K ~20-40%, MATH-500 ~5-15%
(base модель без instruct может быть слабой)
```

**Метрики:**
- **Accuracy** (pass@1): % правильных ответов (ответ извлекается из \boxed{}, проверяется math_verify)
- **Accuracy** (maj@8): генерируем 8 ответов, берём majority vote → более стабильная оценка

---

### ФАЗА 1: Выбор источника данных (3 эксперимента, ~8 часов)

```
Exp 1.1: SFT на трассах DeepSeek-R1 (OpenR1-Math-220K default)
         Данные: ~20K примеров (сэмплируем из 94K, 1 трасса на задачу)
         Метод: Full FT, lr=1e-5, 2 эпохи, prompt masking
         
Exp 1.2: SFT на трассах Qwen3.5-35B-A3B
         Данные: генерируем ответы Q3.5-35B-A3B на тех же 20K промптах
         Метод: Full FT, lr=1e-5, 2 эпохи, prompt masking
         
Exp 1.3: SFT на трассах R1 + KL-constraint с Qwen3.5-35B-A3B
         Данные: те же 20K из OpenR1-Math
         Loss: SFT_loss + α·KL(π_student ∥ π_qwen35)
         Метод: Full FT, lr=1e-5, 2 эпохи
```

**Для Exp 1.2 — генерация трасс:**
```
Qwen3.5-35B-A3B: 35B total, 3B activated (MoE)
В bf16: ~70GB на диске, ~6GB для inference (3B activated)
→ ЛЕГКО влезает в H200, даже одновременно с 4B учеником

Генерация: 20K промптов × ~500 tokens × 1 ответ
Скорость: ~3000 tokens/sec с vLLM → ~3.3M tokens → ~20 минут
```

**Для Exp 1.3 — KL-constraint:**
```python
# В каждом батче:
student_logits = model(input_ids)           # forward
with torch.no_grad():
    teacher_logits = qwen35(input_ids)      # forward через учителя
    
kl_loss = F.kl_div(
    F.log_softmax(student_logits / tau, dim=-1),
    F.softmax(teacher_logits / tau, dim=-1),
    reduction='batchmean'
) * tau**2

total_loss = sft_loss + alpha * kl_loss  # alpha = 0.1-0.5
```

⚠️ **Нюанс для Exp 1.3:** нужно 2 модели в GPU одновременно (4B student + 3B activated teacher = ~14GB). На H200 — без проблем.

✅ **Оба Qwen3.5** → одинаковый vocab (248320) и токенайзер. Logit-based KL работает напрямую.

**Выбор:** Сравниваем Exp 1.1, 1.2, 1.3 на GSM8K test + MATH-500 → берём лучший подход.

---

### ФАЗА 2: Ablation метода обучения (4 эксперимента, ~10 часов)

Используем **лучший датасет** из Фазы 1.

```
Exp 2.1: Full Fine-Tuning
         lr=1e-5, AdamW, cosine schedule, 2 эпохи
         
Exp 2.2: LoRA r=64
         target_modules: все linear слои в Gated DeltaNet и Gated Attention блоках
         (точные имена проверить через model.named_parameters())
         lr=2e-4, AdamW, cosine schedule, 3 эпохи
         
Exp 2.3: DoRA r=64
         То же что LoRA, но с разделением нормы и направления
         lr=2e-4, 3 эпохи
         
Exp 2.4: PiSSA r=64
         SVD-инициализация (top-r singular vectors)
         lr=2e-4, 3 эпохи
```

**Выбор:** Сравниваем → берём лучший метод.

---

### ФАЗА 3: Ablation данных и curriculum (4-6 экспериментов, ~12 часов)

Используем **лучший метод** из Фазы 2.

```
Exp 3.1: Только correct трассы (= наш baseline)

Exp 3.2: Mix correct + incorrect трассы
         50% correct (reward=1) + 50% incorrect (reward=0)
         Label: correct → standard SFT loss
                incorrect → можно: а) игнорировать, б) negative loss, в) DPO

Exp 3.3: Curriculum — от простого к сложному
         Сортируем задачи по сложности (длина трассы как прокси)
         Эпоха 1: задачи с трассами <500 tokens
         Эпоха 2: задачи с трассами 500-2000 tokens
         Эпоха 3: все задачи
         
Exp 3.4: Curriculum — от сложного к простому (anti-curriculum)

Exp 3.5: Random order (= baseline из Фазы 2)

Exp 3.6: Prompt masking vs no prompt masking
         С masking: loss только на ответ (не на промпт)
         Без masking: loss на всю последовательность
```

**Определение сложности для curriculum:**
```python
# Прокси сложности:
# 1. Длина трассы (длиннее = сложнее)
difficulties = [len(tokenizer(trace)["input_ids"]) for trace in traces]
sorted_indices = np.argsort(difficulties)
```

---

### ФАЗА 4: Ablation оптимизатора и scheduler (3-4 эксперимента, ~8 часов)

```
Exp 4.1: AdamW + cosine schedule (baseline)
Exp 4.2: AdamW + linear warmup + constant LR
Exp 4.3: SGD + cosine (проверка: нужен ли Adam?)
Exp 4.4: AdaFactor (memory-efficient alternative)
```

---

## Метрики для всех экспериментов

### Основные (обязательные)
| Метрика | Бенчмарк | Как считать |
|---------|----------|------------|
| **Accuracy (pass@1)** | GSM8K test (1.3K) | greedy decode, extract \boxed{}, math_verify |
| **Accuracy (pass@1)** | MATH-500 | greedy decode, extract \boxed{}, math_verify |
| **Train loss** | — | Логируем каждые 50 шагов |
| **Eval loss** | held-out 5% от train | Логируем каждые 200 шагов |

### Дополнительные (желательные)
| Метрика | Что показывает |
|---------|---------------|
| **maj@8** на MATH-500 | Стабильность модели |
| **Средняя длина ответа** | Не деградирует ли генерация |
| **% format compliance** | Доля ответов с корректным \boxed{} |
| **KL(π_sft ∥ π_base)** | Насколько модель ушла от базы |

### Инструмент оценки

```python
from math_verify import verify_answer
import re

def evaluate(model, dataset):
    correct = 0
    for item in dataset:
        output = model.generate(item["question"], max_tokens=2048)
        match = re.search(r'\\boxed\{(.+?)\}', output)
        if match and verify_answer(match.group(1), item["answer"]):
            correct += 1
    return correct / len(dataset)
```

---

## Гиперпараметры (фиксированные)

```yaml
# Общие для всех:
model: Qwen/Qwen3.5-4B-Base
max_seq_len: 4096          # OpenR1 трассы до 16K, но 4K покрывает ~75%
batch_size: 4              # per device
gradient_accumulation: 4   # effective batch = 16
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
gradient_checkpointing: true

# Full FT:
learning_rate: 1e-5
epochs: 2

# LoRA/DoRA/PiSSA:
learning_rate: 2e-4
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
epochs: 3

# Данные:
num_train_samples: 20000   # из 94K default split
```

---

## Временные оценки (1× H200 140GB)

| Этап | Что делаем | Время |
|------|-----------|-------|
| **Фаза 0** | Baseline eval | ~1 ч |
| **Фаза 1** | Генерация трасс + 3 SFT + eval | ~9 ч |
| **Фаза 2** | 4× SFT + eval | ~10 ч |
| **Фаза 3** | 5-6× SFT + eval | ~15 ч |
| **Фаза 4** | 3-4× SFT + eval | ~10 ч |
| **ИТОГО** | | **~45-55 часов GPU** |

С 1× H200 24/7: **~2.5 дня чистого GPU-time**

---

## Порядок реализации (со мной в чате)

### День 1: Инфраструктура + Baseline (~4ч работы)
```
[ ] Настроить окружение (transformers, trl, peft, vllm, math-verify)
[ ] Скачать модели (Qwen3.5-4B-Base, Qwen3.5-35B-A3B) и датасеты
[ ] Написать eval скрипт (model → GSM8K/MATH-500 → accuracy)
[ ] Прогнать baseline (Exp 0.1)
[ ] Проверить model.named_parameters() для target_modules LoRA
```

### День 2: SFT пайплайн + Фаза 1 (~6ч работы)
```
[ ] SFT training скрипт (prompt masking, wandb logging, checkpointing)
[ ] Подготовка данных: OpenR1-Math → chat format с <think> тегами
[ ] Генерация трасс Q3.5-35B-A3B через vLLM
[ ] Запуск Exp 1.1, 1.2, 1.3
[ ] Eval → выбор лучшего датасета
```

### День 3: Фаза 2 (~4ч работы)
```
[ ] Настроить LoRA/DoRA/PiSSA в скрипте через PEFT
[ ] Запуск Exp 2.1-2.4
[ ] Eval → выбор лучшего метода
```

### День 4-5: Фазы 3-4 (~6ч работы)
```
[ ] Реализовать curriculum (сортировка по длине)
[ ] Реализовать data mix (correct/incorrect)
[ ] Запуск Exp 3.x, 4.x
[ ] Eval → финальные выводы
```

### День 6: Анализ (~3ч работы)
```
[ ] Собрать все результаты → таблицы
[ ] Построить графики (learning curves, ablation bars)
[ ] Написать выводы для proposal/отчёта
```

---

## Чеклист перед стартом

- [ ] H200 доступен
- [ ] Скачать модели Qwen3.5-4B-Base + Qwen3.5-35B-A3B (HF cache)
- [ ] Скачать датасеты: GSM8K, MATH-500, OpenR1-Math-220K
- [ ] Установить: `transformers>=4.51`, `trl`, `peft`, `vllm`, `math-verify`
- [ ] Настроить wandb для логирования
- [ ] Определить GPU-бюджет (сколько часов доступно?)
