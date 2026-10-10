# Проверка диапазона N≤100

**Завершённое синтетическое сравнение; не человеческая валидация.**

**Ровно одна конфигурация прошла полный заявленный допуск.**

Код решения: `SOLE_CONFIGURATION_PASSING_DECLARED_RULE`. Выбрано: `mixed_kg_finer_quadrature`.

Исполнено 548352 quality и 2688 timing случаев. Проверено 123 исходных файлов. Семейство проверок: 89 endpoints.

Пороги для верхних односторонних границ (строгие неравенства): overall <-0.100000, каждое family/stratum <0.500000, CVaR95 <0.500000; здесь regret ×100. Step p95 при каждой заданной размерности <0.250000 с.

## Основная смесь

Числа ниже — нормированный regret ×100, а не доля ошибок найма. Меньше лучше. Неполный результат не заменяется нулём.

| Конфигурация | Роль | SUM | CVaR95 SUM | BEST | ORD | Флаги main |
| --- | --- | --- | --- | --- | --- | --- |
| mixed_kg_finer_quadrature | кандидат | 0.955779 | 7.581659 | 0.081903 | 16.940753 | {} |
| P15_easy | кандидат | 1.054081 | 8.063170 | 0.097300 | 17.877739 | {} |
| linear_evoi_3_boundary_guard | confirmatory baseline | 1.425472 | 10.381790 | 0.091402 | 19.450298 | {} |
| gp_topk_guard | confirmatory baseline | 1.158263 | 8.184110 | 0.111841 | 19.124802 | {} |
| pabbo_rl | описательный контроль | 2.076677 | 13.078360 | 0.203351 | 24.026558 | {} |
| equal_boundary_guard | описательный контроль | 2.313419 | 14.547835 | 0.220323 | 24.683351 | {} |

## Полный допуск

| Кандидат | Качество | Все quality cases | Скорость | Допустим |
| --- | --- | --- | --- | --- |
| mixed_kg_finer_quadrature | PASS | PASS | PASS | PASS |
| P15_easy | FAIL | PASS | PASS | FAIL |

| Кандидат | Baseline | ΔSUM×100 | Односторонняя верхняя×100 | Все quality gates |
| --- | --- | --- | --- | --- |
| mixed_kg_finer_quadrature | linear_evoi_3_boundary_guard | -0.469693 | -0.426948 | PASS |
| mixed_kg_finer_quadrature | gp_topk_guard | -0.202483 | -0.164908 | PASS |
| P15_easy | linear_evoi_3_boundary_guard | -0.371391 | -0.327434 | PASS |
| P15_easy | gp_topk_guard | -0.104181 | -0.063046 | FAIL |

Непройденные ограничения; FAIL интервала не доказывает истинное превышение порога:

| Кандидат | Baseline | Группа | Ограничение | Point×100 | Верхняя×100 |
| --- | --- | --- | --- | --- | --- |
| P15_easy | gp_topk_guard | families | threshold | 0.897015 | 1.084408 |
| P15_easy | gp_topk_guard | gate | overall | -0.104181 | -0.063046 |

## Два кандидата

Порядок разности: `mixed_kg_finer_quadrature` минус `P15_easy`. Среднее×100: -0.098302; двусторонний интервал×100: [-0.129299; -0.067305].

Интервалы среднего используют стратифицированные парные seed-кластеры и приближение Welch-t. Это не distribution-free гарантия покрытия. Для tail bootstrap выполнено 49999 повторов, seed 202610129002; его покрытие также приближённое.

## Отдельный замер времени

| Arm | Метод | Step p95, с | Construction p95, с | Флаги |
| --- | --- | --- | --- | --- |
| timing_n100_d2 | mixed_kg_finer_quadrature | 0.062389 | 0.009944 | {} |
| timing_n100_d2 | P15_easy | 0.022538 | 0.009555 | {} |
| timing_n100_d2 | linear_evoi_3_boundary_guard | 0.062688 | 0.011843 | {} |
| timing_n100_d2 | gp_topk_guard | 0.013171 | 0.011720 | {} |
| timing_n100_d2 | pabbo_rl | 0.027228 | 0.011626 | {} |
| timing_n100_d2 | equal_boundary_guard | 0.000200 | 0.011816 | {} |
| timing_n100_d3 | mixed_kg_finer_quadrature | 0.234384 | 0.011259 | {} |
| timing_n100_d3 | P15_easy | 0.078410 | 0.010616 | {} |
| timing_n100_d3 | linear_evoi_3_boundary_guard | 0.280065 | 0.015782 | {} |
| timing_n100_d3 | gp_topk_guard | 0.018393 | 0.015451 | {} |
| timing_n100_d3 | pabbo_rl | 0.030687 | 0.015317 | {} |
| timing_n100_d3 | equal_boundary_guard | 0.000237 | 0.015569 | {} |

Измерение на данной машине не является SLA. Landmarks остаются описательными; они не добавляются к семейству confirmatory гипотез. Результат относится к заданной синтетической смеси и проверенным конфигурациям, не к мировому оптимуму.

## Область аудита

ROOT_SECOND_IMPLEMENTATION_NOT_INDEPENDENT_AGENT_REVIEW

- Shared SciPy t and NumPy RNG primitives
- Fixed bootstrap index panel for full run, all draws only in small smoke
- No human evaluation, new worlds, optimizer rerun or proof of Python refinement
- No new independent-agent review; previous scalar auditor reused by root

## Проверяемые источники

[Analysis](/Users/arsenij/Documents/ChatGPT/hakaton2/research/frontier_2026/evidence/domain-quality-v1/analysis_domain_v1/analysis.json) · [Аудит](/Users/arsenij/Documents/ChatGPT/hakaton2/research/frontier_2026/formal/output/domain-results-root-audit-v1/audit.json)

SHA256 analysis: `eab45e5f50b60f62456a6eedc4efd5ffa6fab8622e23d968e2dd5065e2191939`.

SHA256 audit: `3198004d36557f314d1dba83ab0c5849e06d72a24c7b0e4512472c9d673e5737`.

Это автоматический перенос уже проверенных чисел. Новые оценки качества,
миры, обучение, подбор порогов и независимая агентская рецензия не выполнялись.
Человеческая валидация: NOT RUN. Итоговая научная интерпретация требует
отдельного прочтения результатов и ограничений.
