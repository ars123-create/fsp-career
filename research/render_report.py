"""Render existing results only; does not rerun or tune the benchmark.

Reuses Matplotlib's bar/errorbar APIs, inspected before implementation.
Run from repository root: python3 -m research.render_report
"""
from pathlib import Path
import csv
import hashlib
import json
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIOS = (
    ("clean", "Нет утечки"),
    ("leakage", "Знание семейств"),
    ("misspecified_leakage", "Ошибка в вероятности"),
    ("shifted_items", "Другая сложность"),
    ("correlated_skills", "Связанные навыки"),
)
POLICIES = (
    ("information_family", "Information gain + семьи", "#939aa7"),
    ("voi_naive", "VOI без семейств", "#cf6b43"),
    ("voi_family", "VOI + семьи", "#157b80"),
    ("pair_voi_family", "VOI с парами + семьи", "#6654a6"),
)


def main():
    root = Path("research/results")
    path = root / "test_v1.json"
    data = json.loads(path.read_text())
    if data["status"] != "COMPLETE":
        raise SystemExit("Refusing to report an incomplete benchmark")
    for name, expected in data["source_sha256"].items():
        source = Path(name)
        if source.is_absolute():
            source = Path("research") / source.name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Source differs from the frozen experiment: {source}")
    rows = data["rows"]
    keys = {(r["scenario"], r["policy"], r["seed"]) for r in rows}
    expected_count = len(data["seeds"]) * len(data["scenarios"]) * len(data["policies"])
    if len(rows) != expected_count or len(keys) != expected_count:
        raise SystemExit("Missing or duplicate benchmark rows")
    metrics = ("regret", "brier", "false_invites", "invitations", "answers",
               "distinct_families", "seconds")
    table = [
        "# Первая отложенная синтетическая проверка",
        "",
        "Автоматически сформировано из test_v1.json. Ни одного реального кандидата.",
        f"Строк: {len(rows)}; рынков в каждом условии: {len(data['seeds'])}.",
        "Regret и Brier: меньше лучше. Ошибочные приглашения — число на рынок, не доля.",
        "Distinct families — число уникальных троек кандидат/навык/семейство среди дополнительных ответов.",
        "",
        "| Условие | Метод | Regret | Brier | Ошиб. приглашения | Все приглашения | Доп. ответы | Семейства | Время, с |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    with (root / "test_v1_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "policy", *metrics])
        for scenario, _ in SCENARIOS:
            for policy in data["policies"]:
                s = data["summary"][scenario][policy]
                writer.writerow([scenario, policy, *[s[k] for k in metrics]])
                table.append("| " + " | ".join([scenario, policy, *[
                    f"{s[k]:.4f}" for k in metrics
                ]]) + " |")
    table += ["", "## Парные описательные интервалы", "",
              "Положительная разница: pair_voi_family имеет меньший regret. Без поправки на множественность.", "",
              "| Условие | Сравнение с | Разница | Нижняя 95% | Верхняя 95% |",
              "|---|---|---:|---:|---:|"]
    for scenario, _ in SCENARIOS:
        for policy, comp in data["summary"][scenario]["paired_comparisons"].items():
            lo, hi = comp["descriptive_bootstrap_95"]
            table.append(f"| {scenario} | {policy} | {comp['regret_reduction']:.4f} | {lo:.4f} | {hi:.4f} |")
    table += ["", "## Хеши исходников", ""]
    for name, digest in data["source_sha256"].items():
        table.append(f"- {Path(name).name}: {digest}")
    table += ["", f"Версия Matplotlib для графика: {matplotlib.__version__}.",
              "Хеширование и проверки полноты результата выполнены при построении отчёта.", ""]
    (root / "test_v1_tables.md").write_text("\n".join(table))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "#fafbf9",
        "axes.facecolor": "#fafbf9",
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(SCENARIOS))
    for idx, (policy, label, color) in enumerate(POLICIES):
        scores = [data["summary"][s][policy]["regret"] for s, _ in SCENARIOS]
        ax.barh(y + (idx - 1.5) * 0.18, scores, height=0.16, label=label, color=color)
    ax.set_yticks(y, [label for _, label in SCENARIOS])
    ax.invert_yaxis()
    ax.set_xlabel("Средний regret — меньше лучше")
    ax.set_title("Качество подбора", loc="left", pad=20)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), frameon=False, fontsize=10)
    means, lowers, uppers = [], [], []
    for s, _ in SCENARIOS:
        c = data["summary"][s]["paired_comparisons"]["voi_family"]
        means.append(c["regret_reduction"])
        lo, hi = c["descriptive_bootstrap_95"]
        lowers.append(means[-1] - lo)
        uppers.append(hi - means[-1])
    bx.errorbar(means, y, xerr=np.array([lowers, uppers]), fmt="o",
                color="#6654a6", capsize=5, linewidth=2)
    bx.axvline(0, color="#343946", lw=1)
    bx.set_yticks(y, [label for _, label in SCENARIOS])
    bx.invert_yaxis()
    bx.set_xlim(-0.15, 0.21)
    bx.set_xlabel("Regret одиночного − regret парного\nПравее нуля: пара лучше")
    bx.set_title("Польза пары проверок", loc="left", pad=20)
    bx.grid(axis="x", alpha=0.2)
    bx.text(0, -0.24, "Все интервалы включают ноль.\nПреимущество пары не подтверждено.",
            transform=bx.transAxes, va="top", fontsize=11, color="#343946")
    fig.suptitle("Проверяем зависимые свидетельства навыков", x=0.03, ha="left", fontsize=19, weight="bold")
    fig.text(0.03, 0.90,
             "Только синтетика • 80 парных рынков на условие • одинаковый бюджет 12 дополнительных ответов",
             fontsize=11, color="#535a65")
    fig.text(0.03, 0.015,
             "Справа: описательные 95% bootstrap-интервалы, 4000 повторов; без поправки на множественность.\n"
             "Источник: research/results/test_v1.json. Результаты не измеряют качество найма реальных людей.",
             fontsize=9, color="#535a65")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.80, bottom=0.30, wspace=0.80)
    fig.savefig(root / "test_v1_summary.png", dpi=160)
    fig.savefig(root / "test_v1_summary.svg")
    plt.close(fig)
    print(f"Verified {len(rows)} rows and frozen source hashes; wrote tables, CSV, PNG and SVG.")


if __name__ == "__main__":
    main()
