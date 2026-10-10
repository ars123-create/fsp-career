"""Render frozen data plus an explicitly post-hoc equal-information baseline.

The post-hoc analysis does not change/relabel the held-out experiment.
Matplotlib 3.10.8 is a report dependency, not needed by the algorithm.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .benchmark import ROOT, hashes, make_case, metrics, summaries, SCENARIOS
from .core import topk

LABELS = {"observed_points":"Точные наблюдения", "bounded_evidence":"Заданные интервалы",
          "polarized_preferences":"Резкие приоритеты", "wrong_priority_constraint":"Неверные приоритеты*",
          "nonlinear_utility":"Неаддитивная полезность*"}


def main():
    data = json.loads((ROOT / "results/heldout_v2.json").read_text())
    frozen = json.loads((ROOT / "protocol.json").read_text())
    assert data["hashes"] == frozen["hashes"] == hashes()
    # Required fairness check added AFTER viewing the main result: same answer,
    # both policies receive that information. Report as exploratory only.
    posthoc = []
    for q in data["queries"]:
        if q["policy"] != "guarded_best_pair":
            continue
        _,lo,hi,_,_,region,utility,_,in_model = make_case(q["seed"],q["scenario"])
        a,b = q["pair"]
        region = region.prefer(a,b) if q["answer"] == 0 else region.prefer(b,a)
        vs=region.vertices()
        nominal=(lo+hi)/2 @ vs.mean(axis=0)
        selected=topk(nominal,5)
        row={"seed":q["seed"],"scenario":q["scenario"],"policy":"updated_mean_same_answer",
             "in_model":in_model,"status":"not_applicable"}
        row.update(metrics(selected,utility,lo,hi,vs,5,nominal))
        posthoc.append(row)
    extra={"status":"POST_HOC_EXPLORATORY", "reason":"Equal information baseline identified after main held-out inspection. Not a preregistered result.",
           "rows":posthoc, "summary":summaries(posthoc)}
    extra_path=ROOT / "results/posthoc_equal_information.json"
    if not extra_path.exists():
        with extra_path.open("x") as f: json.dump(extra,f,indent=2,ensure_ascii=False)
    else:
        assert json.loads(extra_path.read_text()) == extra
    notes=["# Фактические результаты robust top-k v2", "", "7 октября 2026. Синтетика, не качество найма людей.", "",
           "Протокол и хеши зафиксированы до heldout. 500 задач = 5 сценариев × 100 независимых seeds; 18 профилей, 3 признака, k=5. 3500 запусков семи методов и 2000 записей двух стратегий вопроса с/без ограничения. Python 3.14.7, NumPy 2.4.4, SciPy 1.18.1/HiGHS 1.12.0. Отчёт: Matplotlib 3.10.8.", "",
           "**Вывод:** снижение худшего regret не означает улучшение среднего результата. Guarded без вопроса уступает среднему ранжированию по средней реализованной потере. Основной режим приложения не заменён. Полезность уточнений перспективна, но нужна честная проверка с одинаковым объёмом информации и на реальных задачах.", "",
           "## Все методы, все сценарии", "", "Regret ниже лучше; nDCG/пересечение выше лучше. Regret здесь сумма полезностей пяти профилей, максимум 5; в UI показывается 100R/k баллов в среднем на профиль.", "",
           "| Сценарий | Метод | Худший R | Реализованный R | nDCG@5 | Пересечение top-5 | p95 расчёта, с |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        for name,r in data["summary"][s].items():
            notes.append(f"| {LABELS[s]} | {name} | {r['model_regret']:.4f} | {r['realized_regret']:.4f} | {r['ndcg_at_k']:.4f} | {r['overlap_at_k']:.3f} | {r['seconds_p95']:.3f} |")
    notes += ["", "* Два последних сценария нарушают предпосылки гарантии. Отсутствие наблюдаемых нарушений широкого bound в них не подтверждает пригодность модели. Есть отдельный явный контрпример: R=0 внутри неверно заданного W, реальная потеря 1 вне W.", "",
              "## Вопросы и равный объём информации", "", "| Сценарий | Метод | Реализованный R | nDCG@5 |", "|---|---|---:|---:|"]
    for s in SCENARIOS:
        for name,r in data["query_summary"][s].items():
            notes.append(f"| {LABELS[s]} | {name} | {r['realized_regret']:.4f} | {r['ndcg_at_k']:.4f} |")
        r=extra["summary"][s]["updated_mean_same_answer"]
        notes.append(f"| {LABELS[s]} | updated_mean_same_answer **post-hoc** | {r['realized_regret']:.4f} | {r['ndcg_at_k']:.4f} |")
    notes += ["", "Сравнение с mean без вопроса использует разный объём информации и не доказывает превосходство робастного ранжирования. updated_mean_same_answer добавлен после просмотра heldout и помечен post-hoc: это разведочный анализ тех же миров; код/метрики первоначального эксперимента не менялись. Для подтверждающего вывода нужен новый набор с заранее включённым baseline.", "",
              "## Корректность и отрицательные результаты", "",
              "- 58 независимых проверок PASS: углы признаков, полный перебор наборов, сужение W, guard, вырожденность и явный fallback. Тесты не заменяют предпосылки доказательств.",
              f"- Нарушений сертификата в in-model строках: {sum(r['certificate_violation'] for r in data['rows'] if r['in_model'])}; в строках вопросов: {sum(r['certificate_violation'] for r in data['queries'] if r['in_model'])}.",
              f"- Нарушений guard ε=0.05: {sum(r['nominal_loss']>.05+1e-7 for r in data['rows'] if r['policy']=='guarded')}.",
              "- Один основной запуск weights_only / nonlinear_utility / seed 50089 получил bounded с gap 2.41345e−7; его результат оставлен. Две записи вопросов относятся к семьям с хотя бы одной не полностью оптимальной веткой. Не утверждаем optimum для каждого численного запуска.",
              "- До heldout на dev найдено завышенное доверие к default точности HiGHS; tolerance ужесточён, добавлен независимый пересчёт. Dev-файлы сохранены, первая версия кода в archive/dev_v2.",
              "- Guard уменьшил номинальные потери, но не устранил ухудшение среднего истинного результата. Метрики не отбирались по выгоде алгоритму.",
              "- Bootstrap: 2000 парных повторов по независимым задачам, seed 202610071, описательные percentile 95% интервалы; без поправки на множественные сравнения. Все интервалы в heldout_v2.json.", "",
              "![Парные различия](results/heldout_comparison.png)", "",
              "Файлы PROOFS.md и DESIGN.md объясняют цель и происхождение метода; исходный benchmark не редактируется после просмотра. Все практические границы приложения перечислены отдельно в docs/MATCHING_V2.md."]
    (ROOT / "RESULTS.md").write_text("\n".join(notes)+"\n")
    fig, axes=plt.subplots(1,2,figsize=(12.5,5.8),sharey=True,layout="constrained")
    for ax,metric,title in zip(axes,("model_regret","realized_regret"),
                               ("Худший возможный проигрыш","Средний реализованный проигрыш")):
        for j,s in enumerate(SCENARIOS):
            r=data["paired_intervals"][s]["mean"][metric]
            mu=r["guarded_minus_baseline"]*20
            lower,upper=np.array(r["descriptive_95_percentile"])*20
            ax.errorbar(mu,j,xerr=[[max(0,mu-lower)],[max(0,upper-mu)]],fmt="o",capsize=4,
                        color="#28705e" if metric=="model_regret" else "#b84f38",markersize=7)
        ax.axvline(0,color="#555",lw=1)
        ax.grid(axis="x",alpha=.15)
        ax.set_title(title,pad=16)
        ax.set_xlabel("Guarded − обычное среднее, баллы / профиль\n← лучше                   хуже →")
        ax.spines[["top","right","left"]].set_visible(False)
    axes[0].set_yticks(range(5),[LABELS[s] for s in SCENARIOS]); axes[0].invert_yaxis()
    fig.suptitle("Устойчивость улучшилась; средний результат — нет",fontsize=17,fontweight="bold")
    fig.text(.5,-.04,"Синтетика: 100 задач на сценарий; описательные парные 95% интервалы. * Вне предпосылок модели.",ha="center",fontsize=10)
    for ext in ("png","svg"):
        fig.savefig(ROOT/f"results/heldout_comparison.{ext}",dpi=160,bbox_inches="tight")
    print(json.dumps({s:extra['summary'][s]['updated_mean_same_answer']['realized_regret'] for s in SCENARIOS}))


if __name__ == "__main__": main()
