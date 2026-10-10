"""Render the predeclared metrics; no selection of favorable scenarios."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def render(data, result_sha):
    names = {
        "equal": "Равные веса", "app_evoi": "Приложение EVOI", "fixed_swing": "3 фиксированных вопроса",
        "direct_exact": "Истинные веса (контроль)", "direct_round_5pp": "Округление до 5 п.п.",
        "direct_sigma_005": "Шум ввода σ=0.05", "direct_sigma_015": "Шум ввода σ=0.15",
        "direct_sigma_030": "Шум ввода σ=0.30",
    }
    lines = ["# Прямой ввод и уточняющие вопросы: результаты аудита", "",
        f"7 октября 2026. Split: **{data['split']}**, {len(data['cases'])} синтетических задач.",
        "Использован действующий адаптер 0.4.2-dev, включая остановку и ограничения вопросов.",
        "Оценка реальными работодателями, время и сложность ввода **NOT RUN**.", "",
        "Веса прямого ввода известны симулятору, но не адаптивному алгоритму.",
        "Числовые поля передают больше информации, чем бинарные ответы. Это анализ",
        "зависимости от способа ввода и его точности; равное усилие не установлено.", "",
        "## Потеря средней полезности профиля, условные баллы 0–100", "",
        "Меньше лучше. Сценарии публикуются отдельно, общего победителя не выбираем.", "",
        "| Способ | " + " | ".join(data["summary"]) + " |",
        "|---|" + "---:|" * len(data["summary"])]
    for policy, name in names.items():
        lines.append("| " + name + " | " + " | ".join(f"{s['methods'][policy]['mean_regret_points']:.4f}" for s in data["summary"].values()) + " |")
    lines += ["", "## Парные различия EVOI минус контроль", "",
        "Положительное значение означает большую потерю EVOI. Описательные 95%",
        "bootstrap-интервалы по задачам, 2000 повторов; без поправки на множество сравнений.", "",
        "| Сценарий | Контроль | Разница, баллы | Интервал |", "|---|---|---:|---|"]
    for scenario, summary in data["summary"].items():
        for policy, method in summary["methods"].items():
            if policy == "app_evoi":
                continue
            lo, hi = method["app_minus_control_descriptive_ci95"]
            lines.append(f"| {scenario} | {names[policy]} | {method['app_minus_control_points']:+.4f} | [{lo:+.4f}; {hi:+.4f}] |")
    lines += ["", "## Цена вопросов и отрицательные результаты", "",
        "| Сценарий | Среднее ответов EVOI | Без вопроса | Ответов с ухудшением | Итог EVOI хуже равных весов |",
        "|---|---:|---:|---:|---:|"]
    for scenario, summary in data["summary"].items():
        app = summary["methods"]["app_evoi"]
        lines.append(f"| {scenario} | {app['mean_binary_answers']:.2f} | {summary['app_no_question_count']}/{summary['tasks']} | {summary['app_harmful_answer_count']} | {app['worse_than_equal_count']}/{summary['tasks']} |")
    lines += ["", "## Границы интерпретации", "",
        "σ — масштаб исходного искусственного шума; после обрезки/нормирования",
        "ошибка весов другая (RMSE сохранён в JSON). Ни один σ не откалиброван на людях.",
        "Для нелинейности и ошибки навыков даже истинные веса не гарантируют лучшую пятёрку.",
        "Все кандидаты заранее допустимы: этот прогон не проверяет фильтры, грейд или контакты.",
        "Нельзя переносить точность условных тестовых баллов на профессиональные способности.",
        "Нельзя по этому аудиту автоматически выбирать способ ввода для работодателя.", "",
        f"Protocol SHA-256: `{data['protocol_sha256']}`.",
        f"Raw result SHA-256: `{result_sha}`.",
        f"Версии: `{json.dumps(data['versions'], ensure_ascii=False)}`.",
        "Полные входы, ответы, промежуточные подборки, nDCG и RMSE сохранены в JSON.",
        "Дизайн до запуска: [DESIGN.md](DESIGN.md). Исходные v1/v2/v3 не изменены."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    with args.output.open("x") as file:
        file.write(render(json.loads(raw), sha256(raw).hexdigest()))


if __name__ == "__main__":
    main()
