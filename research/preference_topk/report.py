"""Render all frozen results; this script never changes data or algorithm code."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT/"results/heldout_v3.json").read_text())
    dev = json.loads((ROOT/"results/dev_v3.json").read_text())
    protocol = json.loads((ROOT/"protocol.json").read_text())
    actual = {n: sha256((ROOT/n).read_bytes()).hexdigest() for n in protocol["hashes"]}
    if actual != protocol["hashes"] or actual != data["hashes"]:
        raise SystemExit("Source/hash mismatch")
    aggregate = data["comparisons"]["primary_aggregate"]
    gate = {"random_reduction_at_least_10pct": aggregate["random"]["relative_reduction"] >= .1,
            "random_upper_interval_negative": aggregate["random"]["descriptive_95_percentile"][1] < 0,
            "mean_no_worse_than_bald": aggregate["bald"]["difference"] <= 0,
            "mean_no_worse_than_qeubo": aggregate["qeubo"]["difference"] <= 0}
    gate["status"] = "PASS" if all(gate.values()) else "FAIL"
    gate["meaning"] = "Eligible for considering experimental integration only; no real hiring validation or winning guarantee."
    lines = ["# Preference top-k v3: фактическая проверка", "", "7 октября 2026. Только синтетика. В приложение не интегрировано.", "",
             "**Исследовательский критерий допуска: " + gate["status"] + ".** Он разрешает рассмотреть экспериментальную интеграцию, не объявляет качество найма доказанным.", "",
             "1600 новых задач: 8 сценариев × 200 seeds 73000–73199. 18 профилей, 3 навыка, top-5; пять стратегий, одинаковый бюджет до трёх вопросов. Все стратегии используют один и тот же posterior ranker. 32000 строк для бюджетов 0/1/2/3 и 24000 записей вопросов. Seeds разработки 61000–61019 полностью отделены.", "",
             "Код и протокол зафиксированы до проверки: SHA-256 `" + sha256((ROOT/"protocol.json").read_bytes()).hexdigest() + "`. После просмотра теста они не менялись. Python " + data["python"] + ", NumPy " + data["numpy"] + ", SciPy " + data["scipy"] + ".", "",
             "## Заранее выбранный основной результат", "",
             "Средняя потеря полезности одного кандидата после трёх вопросов, с равным весом matched_linear, bounded_evidence, correlated_profiles: 600 независимых задач. Ниже лучше. Для удобства умножаем на 100; это условные баллы модели, не проценты успешного найма.", "",
             "| Метод | Потеря, баллы | Уменьшение потери EVOI относительно метода | Разница EVOI − метод, баллы [описательный 95% интервал] |",
             "|---|---:|---:|---:|"]
    for policy in protocol["policies"]:
        mean = np.mean([data["summary"][s][policy]["3"]["regret_per_candidate"] for s in protocol["primary_scenarios"]])*100
        if policy == "decision_evoi":
            tail = "— | —"
        else:
            c = aggregate[policy]
            lo, hi = c["descriptive_95_percentile"]
            tail = f"{100*c['relative_reduction']:.1f}% | {100*c['difference']:.4f} [{100*lo:.4f}; {100*hi:.4f}]"
        lines.append(f"| {policy} | {mean:.4f} | {tail} |")
    lines += ["", "Преимущество над сильным BALD в этой агрегированной проверке — около 21%, но абсолютная разница всего 0.11 балла условной полезности на шкале 0–100. Это нельзя переносить на работодателей или на исходный режим приложения.", "",
              "## Все сценарии, включая отрицательные", "",
              "Каждая ячейка: средняя потеря в баллах / nDCG@5. Три вопроса, 200 задач в каждой строке. Нелинейность, резкие веса и иной шум проверяют несовпадение модели с миром.", "",
              "| Сценарий | random | uncertainty | BALD | qEUBO | decision-EVOI |",
              "|---|---:|---:|---:|---:|---:|"]
    for scenario, methods in data["summary"].items():
        cells = [f"{methods[p]['3']['regret_per_candidate']*100:.4f} / {methods[p]['3']['ndcg_at_5']:.5f}" for p in protocol["policies"]]
        lines.append("| " + scenario + " | " + " | ".join(cells) + " |")
    negative = sum(q["realized_regret_change"] > 1e-10 for q in data["questions"] if q["policy"] == "decision_evoi")
    total = sum(q["policy"] == "decision_evoi" for q in data["questions"])
    lines += ["", "- EVOI уступил BALD по среднему regret в polarized_weights и nonlinear_geometric. В nonlinear_bottleneck qEUBO лучше EVOI. Универсального превосходства нет.",
              "- Для bounded_evidence и correlated_profiles отдельные интервалы разницы с BALD включают ноль. Основной вывод агрегирован; нельзя утверждать значимое улучшение в каждом сценарии.",
              f"- После {negative} из {total} вопросов decision-EVOI фактическая потеря увеличилась. Неотрицательное ожидаемое VOI не гарантирует пользу каждого ответа.",
              f"- В development основная средняя потеря EVOI была на {-100*dev['comparisons']['primary_aggregate']['bald']['relative_reduction']:.1f}% выше BALD; результат сохранён. Модель после этого не перенастраивалась. Малая dev-выборка не была выдана за итоговый вывод.",
              "- Описательные bootstrap-интервалы: 3000 стратифицированных парных повторов, seed 202610073. Нет поправки на множество сравнений. Все сравнения и бюджеты доступны в JSON, выбор метрики после теста не менялся.", "",
              "## Корректность и стоимость", "",
              "- Минимальный официальный пример SciPy Sobol: PASS до адаптации.",
              "- 53 независимые проверки PASS: перебор наборов и ответов, аналитический centroid, мягкое обновление/противоречия/пропуск, известные контрпримеры.",
              "- Квадратура: 3072 гипотезы в основном исследовании; на 40 dev-задачах сравнение с 12288: та же лучшая пара в 39/40, максимальное изменение её gain 0.0000326. Это ограниченная проверка сходимости, не сертифицированная ошибка интеграла.",
              "- Отдельный одиночный dev-замер полного расчёта: 18 профилей/153 пары ≈0.0064 с; 100 профилей/4950 пар ≈0.20 с. Mac, NumPy/SciPy, OPENBLAS_NUM_THREADS=1. Нет API, p95 и конкурентной нагрузки; это не SLA.",
              "- Тайминги строк benchmark исключают общий первый расчёт вопросов; для сравнения полной задержки не использовать. Весь heldout выполнен за " + f"{data['seconds_total']:.1f}" + " с.", "",
              "## Что остаётся непроверенным", "",
              "Независимая разметка потребностей/кандидатов, полезность самих вопросов для реального работодателя, prior/response-параметры, калибровка экзаменационных оценок, интеграция и сквозной UI. Прототип не меняет грейды и не получает контакты либо реальные профили. Известные методы и их адаптация не объявляются научной новизной.", "",
              "## Воспроизведение", "", "```sh",
              ".venv-matching/bin/python -m research.preference_topk.smoke_qmc",
              ".venv-matching/bin/python -m pytest research/preference_topk/test_core.py -q",
              "OPENBLAS_NUM_THREADS=1 .venv-matching/bin/python -m research.preference_topk.benchmark --split heldout --output /tmp/fsp-preference-reproduction.json",
              "```", "",
              "Первичные источники, лицензии, границы адаптации: [DESIGN.md](DESIGN.md). Условные свойства: [PROOFS.md](PROOFS.md)."]
    with (ROOT/"RESULTS.md").open("x") as f:
        f.write("\n".join(lines)+"\n")
    with (ROOT/"results/gate.json").open("x") as f:
        json.dump(gate, f, ensure_ascii=False, indent=2)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
