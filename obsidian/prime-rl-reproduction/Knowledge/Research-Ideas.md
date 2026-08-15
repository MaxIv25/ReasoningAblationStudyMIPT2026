---
type: synthesis
status: hypotheses
updated: 12-08-2026
tags: [prime, hypotheses, dpo-z, process-reward, guided-search]
---
# Четыре авторские гипотезы

## Статус реализации — 12-08-2026

Все четыре идеи присутствуют в коде за независимыми flags, но ещё не имеют experiment evidence:

- DPO-Z: корректная `beta * logmeanexp(R / beta)` leave-one-out baseline доступна и в ordinary GRPO, и в PRIME; дополнительное centering не применяется.
- Gradual process reward: `constant`, `linear` и `reliability_gate` schedules.
- Z(x)-calibration: реализован gauge-complete prompt intercept с absolute BCE; trainer также поддерживает optional within-prompt ranking.
- PRM-guided search: candidate-pool `topk` требует явного согласия на biased/off-policy update; stochastic mode возвращает proposal probabilities и Horvitz–Thompson weights. Это ещё не token-level beam implementation.

Наличие кода не меняет оценки ниже и не считается подтверждением гипотез.

> [!warning]
> Ни одна из идей пока не имеет валидного ablation result. Они заморожены до faithful PRIME reproduction.

## Общая теоретическая мотивация

В DPO reward можно записать как

$$
r(x,y)=\beta_{\mathrm{DPO}}\log
\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta_{\mathrm{DPO}}\log Z(x).
$$

Первое слагаемое похоже на implicit process reward в PRIME, а второе мотивирует идеи `DPO-Z baseline` и `Z(x)`-calibration. Однако здесь важны три различия:

- reward functions, различающиеся только на prompt-dependent constant $f(x)$, задают одинаковые within-prompt preferences; DPO выбирает каноническую parameterization, фактически фиксируя эту gauge freedom;
- в implicit PRM масштаб

  $$
  q_\phi(x,y)=\beta_{\mathrm{PRM}}\log
  \frac{\pi_\phi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
  $$

  уже задаёт нормированный представитель, для которого partition term равен нулю;
- $\beta_{\mathrm{DPO}}$ — KL temperature в DPO derivation, а $\beta_{\mathrm{PRM}}$ — масштаб implicit reward/logit. Их нельзя автоматически отождествлять. В original PRIME actor KL равен нулю, поэтому DPO optimal-policy derivation не переносится на actor update буквально.

## 1. DPO-Z advantage baseline

Заявленная soft-value формула:

$$
V_\beta(x)=\beta\log\mathbb{E}_{y\sim\pi_{\mathrm{ref}}}
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

### Найденная ошибка реализации

- Poster: `β [logsumexp(r/β) - log K]`.
- Legacy code: `[logsumexp(βr) - log K] / β`.

При `β=.05` первая формула близка к max reward, вторая — почти к arithmetic mean. Это разные algorithms.

### Другие проблемы

- Rollouts идут из current policy, а expectation задано по `π_ref`; importance correction нет.
- Baseline включает собственный `r_i`, поэтому не является строгим action-independent control variate; корректнее leave-one-out soft baseline.
- В legacy-коде DPO-Z применялся только к outcome component; новая реализация использует один явно выбранный baseline для outcome и process sources.
- Для binary reward и малого `β` значение почти определяется наличием одного correct response.
- Prompt-constant `log Z(x)` в обычном within-prompt relative advantage и так сокращается.
- Если прибавить один и тот же $b_Z(x)$ ко всем rewards, а затем применить RLOO, он сократится точно. Поэтому DPO-Z должен **заменять** RLOO baseline или менять prompt weighting, а не добавляться перед неизменным RLOO.

### Корректная проверяемая версия

$$
b_Z(x)=\beta_{\mathrm{DPO}}
\log\mathbb{E}_{y\sim\pi_{\mathrm{ref}}}
\exp\left(\frac{R(x,y)}{\beta_{\mathrm{DPO}}}\right),
\qquad
A_i=R_i-b_Z^{(-i)}(x).
$$

Требования к реализации:

- samples из `π_ref` либо явные importance weights для samples из actor policy;
- leave-one-out или независимая оценка относительно текущего action;
- `stop_gradient` через baseline;
- формула `β * logmeanexp(R / β)`, а не `logmeanexp(βR) / β`;
- отдельное сравнение variance, stability и sample efficiency против standard RLOO.

Для binary outcome reward при $p_{\mathrm{ref}}(x)=P_{\pi_{\mathrm{ref}}}(R=1\mid x)$:

$$
b_Z(x)=\beta\log\left[(1-p_{\mathrm{ref}}(x))
+p_{\mathrm{ref}}(x)e^{1/\beta}\right].
$$

Это показывает, что при малом $\beta$ baseline становится near-max и risk-sensitive, а не просто оценкой средней сложности prompt.

### Вердикт

Текущая legacy-версия: **4/10**. При правильной реализации: **6–7/10** как новый risk-sensitive advantage baseline. Идея теоретически мотивирована и достойна ablation, но её не следует называть восстановлением обязательного «пропущенного члена DPO»: prompt constant неидентифицируем и сокращается в обычных relative advantages.

## 2. Постепенное включение process reward

$$
A_t=R_{\mathrm{outcome}}+\alpha(s)R_{\mathrm{process}}(t),
\qquad \alpha:0\rightarrow1.
$$

### Почему идея разумна

Ранний PRM signal может быть шумным. Явная schedule ограничивает его влияние на actor, пока PRM не научился различать correct/incorrect rollouts.

### Ограничения

- В original `single-forward` первый process reward уже равен нулю, так как PRM и reference идентичны.
- Linear schedule по step не измеряет реальное качество PRM.
- Final advantage whitening изменяет фактический вклад `α` в gradient.
- Старый curriculum run не даёт полностью аудируемого подтверждения.

### Вердикт

Самая перспективная из исходных трёх: **8/10** как practical hypothesis. Более сильная версия — reliability-gated `α` по held-out PRM NLL/AUC/calibration или ranking stability; linear warmup должен быть baseline.

## 3. Z(x)-calibrated PRM loss

Для

$$
q_\phi(x,y)=\beta_{\mathrm{PRM}}
\log\frac{\pi_\phi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

выполняется

$$
\mathbb{E}_{y\sim\pi_{\mathrm{ref}}}
\exp\left(\frac{q_\phi(x,y)}{\beta_{\mathrm{PRM}}}\right)
=\sum_y \pi_\phi(y\mid x)=1.
$$

Значит, `log mean exp(q)` при правильном sampling из reference сходится к нулю. Prompt-dependent additive constant DPO reward не идентифицируется только из policy ratio.

Legacy code оценивает expression по samples из policy, а не reference. Это не partition function; если policy близка к PRM, величина ближе к Rényi-2 divergence. Detached logit shift может быть эвристикой, но не `Z(x)` calibration.

Также positive shift упрощает positive labels и усложняет negative labels; автоматический focus на сложных prompts из этого не следует.

### Буквальный вердикт

Восстановление ненулевого «истинного $Z(x)$» только из policy ratio: **2/10** даже при безошибочной реализации. Здесь проблема не инженерная, а фундаментальная: prompt-dependent additive constant не идентифицируется из pairwise preferences или нормированного policy ratio.

### Более сильная переформулировка: gauge-complete PRM

Вместо вычисления `Z(x)` из ratio ввести явный prompt intercept:

$$
s_{\phi,\psi}(x,y)=
a\log\frac{\pi_\phi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+b_\psi(x).
$$

- ratio term отвечает за различия между responses одного prompt;
- $b_\psi(x)$ моделирует абсолютную difficulty/calibration и отсутствующую prompt-dependent gauge;
- scale $a$ обучается или настраивается отдельно и не обязан совпадать ни с $\beta_{\mathrm{DPO}}$, ни с $\beta_{\mathrm{PRM}}$.

Чтобы intercept не выучил только class prior или difficulty, loss должен сочетать:

1. absolute correctness BCE/calibration loss для $s_{\phi,\psi}(x,y)$;
2. within-prompt ranking loss, в котором $b_\psi(x)$ сокращается и response discrimination остаётся обязательной.

Перспективность такой постановки: **6–7/10**. Это уже отдельная calibration hypothesis, а не точное восстановление DPO partition function.

> [!important] Где использовать intercept
> $b_\psi(x)$ не даёт token-level credit. В actor-side RLOO он сокращается, поэтому его разумнее проверять для PRM calibration, prompt filtering/weighting или определения надёжности process reward, но не безусловно прибавлять к каждому token reward.

## 4. Online implicit-PRM-guided rollout search

### Мотивация

В PRIME доступен token-level implicit process reward:

$$
r_t^{\mathrm{PRM}}
=
\beta_{\mathrm{PRM}}
\left[
\log \pi_\phi(y_t\mid y_{<t},x)
-
\log \pi_{\mathrm{ref}}(y_t\mid y_{<t},x)
\right].
$$

Его сумма на prefix равна

$$
R_{\leq t}^{\mathrm{PRM}}
=
\beta_{\mathrm{PRM}}
\log
\frac{\pi_\phi(y_{\leq t}\mid x)}
{\pi_{\mathrm{ref}}(y_{\leq t}\mid x)}.
$$

Поэтому PRM технически можно использовать не только после rollout для advantage estimation, но и во время генерации для выбора перспективных prefixes.

### Что уже известно и где возможна новизна

PRM-guided beam/tree search, Best-of-N и reward-guided decoding уже являются отдельным направлением; сама идея использовать process score при поиске не нова. В original PRIME этого нет: rollouts семплируются policy, а PRM оценивает их после генерации.

Потенциально интересная постановка проекта — **online co-evolving implicit PRM как guided proposal policy для следующих training rollouts**, вместе с reliability gating и корректным учётом behavior policy.

### Возможные варианты

1. `Best-of-M`: независимо сгенерировать $M>8$ полных responses и оставить восемь с максимальным PRM score.
2. `Chunk-level stochastic beam`: разветвлять continuations каждые 128–256 tokens или на границах reasoning steps и сохранять diverse beam ширины 8.
3. `Token-level deterministic beam`: ранжировать каждый следующий token.

Третий вариант наименее предпочтителен: шумный локальный reward может преждевременно удалить перспективную reasoning branch, а deterministic beam резко снижает diversity.

Для beam score нельзя использовать только log-ratio PRM. Более безопасная форма:

$$
S(y_{\leq t})
=
\operatorname{norm}\log\pi_\theta(y_{\leq t}\mid x)
+
\lambda\operatorname{norm}R_{\leq t}^{\mathrm{PRM}},
$$

иначе search может предпочитать редкие или странные tokens, которым `π_ref` присваивает особенно низкую вероятность.

### Предпочтительная формулировка: guided stochastic proposal

Вместо hard top-k beam определить явную behavior policy:

$$
q_\lambda(y_t\mid s_t)
\propto
\pi_\theta(y_t\mid s_t)
\exp\left(
\lambda\,\operatorname{clip}
\left[r_t^{\mathrm{PRM}}\right]
\right).
$$

Плюсы этой формы:

- сохраняется stochastic exploration;
- normalized `log q_\lambda` можно вычислить и записать;
- behavior policy известна, поэтому возможны importance correction или clipped off-policy weights;
- $\lambda$ естественно связать с gradual process weighting: начинать с нуля и повышать только после достижения PRM заданной held-out AUC/calibration.

При инициализации `PRM == reference`, поэтому $r_t^{\mathrm{PRM}}=0$ и guided proposal совпадает с actor policy.

### Почему нельзя просто обучать PRIME на восьми лучших beams

- После selection trajectories уже не распределены как $\pi_{\theta_{\mathrm{old}}}$; стандартный PPO/GRPO ratio становится неверным.
- Top beams могут быть почти одинаковыми, уменьшая exploration и effective sample size.
- Если все восемь responses станут correct или получат одинаковый reward, RLOO advantage обнулится, а official accuracy filter удалит group.
- Возникает feedback loop: PRM выбирает удобные для себя trajectories, затем переобучается на собственной селекции.
- False-positive PRM prefix может вытеснить правильные branches; final outcome verifier остаётся обязательным.
- Raw implicit reward может быть плохо откалиброван для сравнения prefixes разной длины и разных trajectories.

### Минимальный проверяемый ablation

До изменения training loop заморозить actor/reference/PRM snapshot и при одинаковом бюджете generated tokens сравнить:

1. independent sampling;
2. `Best-of-M`;
3. stochastic chunk beam;
4. guided sampling из $q_\lambda$.

Основные метрики:

- verified `pass@1`, `pass@8` и oracle `pass@M`;
- число уникальных responses и unique final answers;
- PRM AUC/calibration и false-positive rate в верхнем reward quantile;
- корреляция reward с response length;
- accuracy на единицу generation compute.

Только после положительного frozen-search результата переходить к training. Первый безопасный вариант — mixture обычных on-policy и guided rollouts, причём verified-correct guided trajectories использовать через явно отделённый auxiliary distillation objective. Прямой PRIME update на selected top-8 без behavior correction не использовать.

### Вердикт

- Широкая идея `PRM-guided beam search`: **3/10 по новизне**.
- Online implicit-PRM-guided training rollouts: **6/10**.
- Reliability-gated stochastic proposal с известной behavior policy и off-policy correction: **7/10**.

Исследовательский вопрос:

> Может ли online implicit PRM улучшить exploration и sample efficiency PRIME, если использовать его как постепенно включаемую stochastic proposal policy, сохраняя корректность policy update?

## Единая исследовательская линия

Наиболее цельная постановка после reproduction — **Calibrated PRIME**:

1. explicit prompt intercept для абсолютной PRM calibration;
2. reliability-gated включение process reward в actor objective;
3. reliability-gated PRM guidance для exploration, только если frozen-search ablation подтверждает качество ranking.

`DPO-Z advantage baseline` лучше оставить отдельной actor-side гипотезой: он проверяет risk-sensitive control variate, а не решает ту же задачу, что PRM calibration.

## Приоритет после reproduction

1. Linear и reliability-gated process weighting.
2. Frozen implicit-PRM-guided search ablation без изменения training distribution.
3. Guided stochastic proposal с behavior correction, только после положительного результата пункта 2.
4. Corrected leave-one-out DPO-Z с exact toy-distribution и variance tests против RLOO.
5. Gauge-complete PRM с prompt intercept; буквальный `Z(x)` loss не запускать.

## Связанные источники

- [[Papers/Process Reinforcement through Implicit Rewards|PRIME]]
- [[Papers/Free Process Rewards without Process Labels|Implicit PRM]]
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Self-Evaluation Guided Beam Search for Reasoning](https://arxiv.org/abs/2305.00633)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
- [Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning](https://arxiv.org/abs/2410.08146)
- [Linking Process to Outcome: Conditional Reward Modeling for LLM Reasoning](https://arxiv.org/abs/2509.26578)
