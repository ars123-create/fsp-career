"""Versioned, deterministic task families with executable reference answers.

These are demonstration rubrics, not empirically calibrated grade instruments.
No untrusted candidate code is executed. Family dependence is visible in evidence.
"""
import random
from collections import defaultdict
from app.practical_bank import VERSION, build_practical, correct as practical_correct
from app.practical_bank_v2 import VERSION as VERSION_V2, build_practical as build_practical_v2

SPECIALIZATIONS = {"backend": "Python backend", "data": "Аналитика данных / SQL"}
SKILLS = {"backend": ["Python", "SQL", "API"], "data": ["SQL", "Аналитика", "Качество данных"]}
RUBRIC_V1 = {"version": "2026-10-v1", "pass_score": 70, "min_skill_score": 50,
          "notice": "Демонстрационная рубрика. Сложность и соответствие грейду ещё требуют независимой калибровки."}
RUBRIC_V2 = {"version": VERSION_V2, "pass_score": 75, "min_skill_score": 50,
          "min_practical_skill_score": 50,
          "notice": "Демонстрационная рубрика. Эталоны проверены вычислением; сложность и границы грейдов ещё не откалиброваны на людях."}
RUBRIC = {**RUBRIC_V2, "version": VERSION}


def rubric_for(version):
    if version == RUBRIC_V1["version"]:
        return dict(RUBRIC_V1)
    if version == VERSION_V2:
        return dict(RUBRIC_V2)
    if version == VERSION:
        return dict(RUBRIC)
    raise ValueError("Unknown assessment rubric version")


def question(rng, family, skill, title, prompt, correct, alternatives, explanation, code=""):
    answers = list(dict.fromkeys([str(correct), *map(str, alternatives)]))
    if len(answers) < 4:
        answers += [str(1000 + x) for x in range(4 - len(answers))]
    answers = answers[:4]
    rng.shuffle(answers)
    return {"id": family, "family": family, "skill": skill, "title": title, "prompt": prompt,
            "code": code, "choices": answers, "_correct": answers.index(str(correct)), "_explanation": explanation}


def build_test_v1(specialization: str, grade: str, seed: int, kind="grade") -> list[dict]:
    rng = random.Random(seed)
    a, b, c = rng.sample(range(11, 80), 3)
    hard = grade == "Middle"
    n = rng.randint(4, 9)
    tasks = []
    q = lambda *args, **kwargs: question(rng, *args, **kwargs)
    # Shared SQL families have the same meaning across both specialties.
    values = rng.choices([a, b, c, a + b, b + c], k=rng.randint(3, 8)) + [None] * rng.randint(1, 3)
    rng.shuffle(values)
    if hard:
        correct = len({x for x in values if x is not None})
        prompt = "В столбце value строки указаны ниже. Что вернёт COUNT(DISTINCT value)?"
        code = f"values: {values}\nSELECT COUNT(DISTINCT value) FROM events;"
    else:
        correct = len([x for x in values if x is not None])
        prompt = "В столбце value есть NULL. Какой результат вернёт COUNT(value)?"
        code = f"values: {values}\nSELECT COUNT(value) FROM events;"
    tasks.append(q("sql_null_distinct", "SQL", "Агрегация без сюрпризов", prompt,
                   correct, [0, 1, 3 if correct != 3 else 4, 5],
                   "COUNT(value) пропускает NULL; DISTINCT дополнительно удаляет дубликаты.", code))
    left = rng.randint(3, 7)
    matches = rng.randint(2, left - 1)
    extra = rng.randint(2, 5)
    correct = matches + extra + (left - matches) if hard else matches + extra
    join = "LEFT JOIN" if hard else "INNER JOIN"
    tasks.append(q("sql_join_cardinality", "SQL", "Кардинальность соединения",
                   f"У нас {left} пользователей. У {matches} из них по одному заказу; у одного из этих же пользователей ещё {extra} заказов. У остальных заказов нет. Сколько строк даст {join} пользователей с заказами по user_id?",
                   correct, [left, matches, left * (matches + extra), 0],
                   f"Каждому заказу соответствует строка; {'LEFT JOIN сохраняет также пользователей без заказов.' if hard else 'INNER JOIN исключает пользователей без заказов.'}",
                   f"SELECT u.id, o.id FROM users u\n{join} orders o ON u.id = o.user_id;"))
    if specialization == "backend":
        nums = rng.choices([a, b, c], k=rng.randint(3, 8))
        if hard:
            actual = sum({v: i for i, v in enumerate(nums)}.values())
            code = f"values = {nums}\npositions = {{v: i for i, v in enumerate(values)}}\nprint(sum(positions.values()))"
            explain = "Повторный ключ перезаписывается последним индексом; суммируются значения словаря."
        else:
            actual = sum(v for v in nums if v > b)
            code = f"values = {nums}\nprint(sum(v for v in values if v > {b}))"
            explain = "Сначала отбираются значения по условию, затем суммируются; дубликаты сохраняются."
        tasks.append(q("python_collections", "Python", "Коллекции и преобразования",
                       "Что выведет фрагмент Python?", actual, [actual + 1, actual + a, 0],
                       explain, code))
        initial = [a, b]
        if hard:
            actual = a + c + b
            code = f"a = [[{a}], [{b}]]\nb = a.copy()\nb[0].append({c})\nprint(sum(sum(row) for row in a))"
            explain = "Поверхностная копия списка сохраняет ссылки на вложенные списки."
        else:
            actual = a + b
            code = f"a = {initial}\nb = a.copy()\nb.append({c})\nprint(sum(a))"
            explain = "Добавление элемента в отдельную копию внешнего списка не изменяет исходный список."
        tasks.append(q("python_mutability", "Python", "Изменяемость и ссылки",
                       "Какое число будет напечатано?", actual, [actual + c, 0, a],
                       explain, code))
        repeats = rng.randint(2, 6)
        balance = rng.randint(100, 200) * 100
        amount = rng.randint(10, 40) * 100
        actual = balance - amount
        tasks.append(q("api_idempotency", "API", "Повтор запроса",
                       f"Баланс {balance} ₽. Клиент отправил списание {amount} ₽ и повторил его {repeats} раз с тем же idempotency key. Сервер атомарно сохраняет результат первого запроса по ключу. {'Первый ответ потерян, все повторы приходят параллельно. ' if hard else ''}Каков итоговый баланс?",
                       actual, [balance - amount * (repeats + 1), balance, 0],
                       "Атомарная фиксация ключа и результата означает одно списание при повторах."))
        stock = rng.randint(5, 20)
        take = rng.randint(2, stock - 1)
        concurrent = rng.randint(2, 5)
        if hard:
            actual = stock - take
            prompt = f"Остаток {stock}. {concurrent} клиента прочитали version=7. Каждый выполняет условный UPDATE: уменьшить остаток на {take}, увеличить version на 1, только если version=7. Запросы атомарны; неуспешные не повторяются. Какой остаток?"
            code = f"UPDATE stock SET count=count-{take}, version=version+1\nWHERE id=1 AND version=7;"
            explain = "Первый UPDATE изменит версию; остальные условия не выполнятся."
        else:
            actual = concurrent + 1
            prompt = f"Клиент делает один запрос. При временной ошибке разрешено ещё {concurrent} повторных попытки. Сколько запросов максимум уйдёт?"
            code = ""
            explain = "Число запросов — одна исходная попытка плюс разрешённые повторы."
        tasks.append(q("api_concurrency", "API", "Сбои и согласованность", prompt,
                       actual, [stock, stock - take * concurrent, concurrent + 2, 0], explain, code))
    else:
        observations = [a, a, a, b, c] if hard else [a, b, c]
        actual = sorted(observations)[len(observations) // 2]
        tasks.append(q("analytics_median", "Аналитика", "Устойчивая статистика",
                       f"Найдите медиану значений: {observations}.", actual,
                       [sum(observations), max(observations) + 1, 0],
                       "Медиана нечётного набора — центральное значение отсортированного ряда."))
        visits = rng.randint(5, 15) * 100
        rate = rng.choice([10, 20, 25, 40])
        buyers = visits * rate // 100
        if hard:
            visits2 = visits * 2
            rate2 = rng.choice([5, 15, 30])
            actual = round((buyers + visits2 * rate2 // 100) / (visits + visits2) * 100, 2)
            prompt = f"Канал A: {visits} визитов, {buyers} покупок. Канал B: {visits2} визитов, {visits2 * rate2 // 100} покупок. Общая конверсия в процентах, до двух знаков?"
        else:
            actual = float(rate)
            prompt = f"Из {visits} визитов получилось {buyers} покупок. Конверсия в процентах?"
        tasks.append(q("analytics_conversion", "Аналитика", "Знаменатель метрики", prompt,
                       actual, [round(actual + 5, 2), round(actual / 2, 2), 100.0],
                       "Конверсия — сумма покупок, делённая на сумму визитов; среднее процентов каналов может быть неверным."))
        rows = rng.randint(100, 200)
        missing = rng.randint(2, 9)
        duplicates = rng.randint(3, 8)
        actual = rows - missing - duplicates
        tasks.append(q("quality_dedup", "Качество данных", "Очистка данных",
                       f"В таблице {rows} строк. Удалили {missing} строк с пустым ключом. Среди оставшихся есть {duplicates} лишних дублирующих строк. {'Уникальность определяется составным ключом (source, event_id). ' if hard else ''}Сколько строк осталось после удаления лишних дублей?",
                       actual, [rows, rows - missing, rows - duplicates, 0],
                       "Лишние дубли уже подсчитаны после удаления пустых ключей, их нужно вычесть один раз."))
        first = rng.randint(10, 30)
        late = rng.randint(2, 8)
        if hard:
            actual = first
            prompt = f"В потоке {first} уникальных event_id, затем {late} повторных доставок уже полученных событий. Загрузка идемпотентна по event_id. Сколько событий учтёт сумма COUNT после upsert?"
            explain = "Повторная доставка с тем же ключом обновляет запись, не создавая новое событие."
        else:
            actual = first - late
            prompt = f"Из {first} строк {late} не прошли обязательную проверку схемы и отправлены в карантин. Сколько строк попадёт в основную таблицу?"
            explain = "Строки, нарушающие схему, не входят в основную таблицу."
        tasks.append(q("quality_pipeline", "Качество данных", "Надёжность загрузки", prompt,
                       actual, [first + late, first * late, 0], explain))
    rng.shuffle(tasks)
    if kind == "challenge":
        chosen = {}
        for task in tasks:
            chosen.setdefault(task["skill"], task)
        tasks = list(chosen.values())
    return tasks


def build_test(specialization: str, grade: str, seed: int, kind="grade", *, version=VERSION) -> list[dict]:
    rubric_for(version)
    if version == RUBRIC_V1["version"]:
        return build_test_v1(specialization, grade, seed, kind)
    tasks = (build_practical_v2 if version == VERSION_V2 else build_practical)(specialization, grade, seed)
    if kind == "grade":
        tasks += build_test_v1(specialization, grade, seed)
    elif kind != "challenge":
        raise ValueError("Unknown assessment kind")
    random.Random(seed ^ 0xF5F2026).shuffle(tasks)
    if kind == "challenge":
        chosen = {}
        for task in tasks:
            chosen.setdefault(task["skill"], task)
        return list(chosen.values())
    return tasks


def answer_correct(task, value):
    if task.get("answer_type") == "fields":
        return practical_correct(task, value)
    return type(value) is int and value == task["_correct"]


def public_questions(tasks: list, completed=False, answers=None) -> list:
    output = []
    for task in tasks:
        public = {k: v for k, v in task.items() if not k.startswith("_")}
        public.setdefault("answer_type", "choice")
        if completed:
            key = "correct_values" if task.get("answer_type") == "fields" else "correct_choice"
            public.update({key: task["_correct"], "explanation": task["_explanation"]})
            if answers is not None:
                public["is_correct"] = answer_correct(task, answers.get(task["id"]))
        output.append(public)
    return output


def score_test(tasks: list, answers: dict, *, rubric_version=None) -> dict:
    if not tasks:
        raise ValueError("An assessment must contain questions")
    rubric = rubric_for(rubric_version or (VERSION if any(q.get("answer_type") == "fields" for q in tasks) else RUBRIC_V1["version"]))
    by_skill = defaultdict(list)
    practical = defaultdict(list)
    correct = 0
    for task in tasks:
        ok = answer_correct(task, answers.get(task["id"]))
        by_skill[task["skill"]].append(int(ok))
        if task.get("answer_type") == "fields":
            practical[task["skill"]].append(int(ok))
        correct += ok
    skills = {k: round(100 * sum(v) / len(v), 1) for k, v in by_skill.items()}
    practical_skills = {k: round(100 * sum(v) / len(v), 1) for k, v in practical.items()}
    score = round(100 * correct / len(tasks), 1)
    practical_ok = all(practical_skills.get(k, 0) >= rubric.get("min_practical_skill_score", 0) for k in skills)
    return {"score": score, "correct": correct, "total": len(tasks), "skills": skills,
            "practical_skills": practical_skills,
            "passed": score >= rubric["pass_score"] and min(skills.values()) >= rubric["min_skill_score"] and practical_ok,
            "rubric": rubric}
