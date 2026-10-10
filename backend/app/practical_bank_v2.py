"""Bounded practical cases. No candidate code, SQL, expression or pickle execution.

Generated inputs and numeric outputs check specific demonstrated operations,
not a calibrated professional grade. Decimal is the unchanged Python stdlib.
"""
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import random
import re

VERSION = "2026-10-v2"


def number(text, places=0):
    if not isinstance(text, str) or len(text) > 40:
        raise ValueError("Введите число длиной до 40 символов.")
    text = text.strip().replace(",", ".")
    if not re.fullmatch(r"[+-]?[0-9]{1,12}(?:\.[0-9]{1,6})?", text):
        raise ValueError("Введите число без единиц, формул и разделителей тысяч; запятая и точка допустимы.")
    value = Decimal(text)
    if places == 0 and value != value.to_integral_value():
        raise ValueError("В этом поле требуется целое число.")
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def validate_values(task, values):
    if not isinstance(values, dict) or set(values) != {f["name"] for f in task["fields"]}:
        raise ValueError("Заполните все поля этого задания.")
    return {f["name"]: str(number(values[f["name"]], f["places"])) for f in task["fields"]}


def correct(task, values):
    try:
        normalized = validate_values(task, values)
        return all(Decimal(normalized[k]) == Decimal(v) for k, v in task["_correct"].items())
    except (ValueError, TypeError, KeyError):
        return False


def table(name, columns, rows):
    return {"name": name, "columns": columns, "rows": [list(row) for row in rows]}


def task(family, skill, title, prompt, answers, labels, explanation, *, code="", tables=(), places=None):
    places = places or {}
    fields = [{"name": k, "label": labels[k], "places": places.get(k, 0)} for k in answers]
    expected = {k: str(Decimal(str(v)).quantize(Decimal(1).scaleb(-places.get(k, 0)), rounding=ROUND_HALF_UP)) for k, v in answers.items()}
    return {"id": family, "family": family, "skill": skill, "title": title,
            "prompt": prompt, "code": code, "tables": list(tables), "answer_type": "fields",
            "fields": fields, "choices": [], "_correct": expected, "_explanation": explanation}


def build_practical(specialization, grade, seed):
    if specialization not in ("backend", "data") or grade not in ("Junior", "Middle"):
        raise ValueError("Unknown assessment specialization or grade")
    rng = random.Random(seed ^ 0x26A55E55)
    hard = grade == "Middle"
    out = []
    # The production oracle is ordinary Python; tests independently execute SQL.
    rows = [[i+1, rng.randint(1, 3), rng.choice([20, 30, 40, 60, None]), rng.choice(["paid", "paid", "cancelled"])] for i in range(9)]
    threshold = rng.choice([40, 60, 80])
    totals = defaultdict(int)
    for _, customer, amount, state in rows:
        if state == "paid" and amount is not None:
            totals[customer] += amount
    selected = [v for v in totals.values() if v >= threshold] if hard else list(totals.values())
    # NULL-only groups contribute zero to the sum, but COUNT(*) still counts them.
    customers = {r[1] for r in rows if r[3] == "paid"}
    count = len(selected) if hard else len(customers)
    having = f"\n  HAVING SUM(amount) >= {threshold}" if hard else ""
    code = "WITH report AS (\n  SELECT customer_id, SUM(amount) AS total\n  FROM orders WHERE state = 'paid'\n  GROUP BY customer_id" + having + "\n)\nSELECT COUNT(*) AS customers, COALESCE(SUM(total), 0) AS revenue\nFROM report;"
    out.append(task("sql_group_report", "SQL", "Отчёт по оплатам", "Выполните запрос мысленно на приведённой таблице orders. NULL — отсутствие значения. Введите два значения единственной строки результата.",
        {"customers": count, "revenue": sum(selected)}, {"customers":"customers — клиентов", "revenue":"revenue — сумма"},
        "WHERE отбирает оплаченные строки до группировки. SUM пропускает NULL; HAVING, если задан, применяется к сумме группы. COALESCE заменяет пустую сумму нулём.", code=code,
        tables=[table("orders", ["id", "customer_id", "amount", "state"], rows)]))
    rows = [[i+1, rng.randint(1, 3), rng.randint(1, 5), rng.randint(2, 20), rng.choice(["active", "active", "deleted"]) if hard else "active"] for i in range(8)]
    latest = {}
    for row in rows:
        if row[1] not in latest or (row[2], row[0]) > (latest[row[1]][2], latest[row[1]][0]):
            latest[row[1]] = row
    live = [r for r in latest.values() if r[4] == "active"]
    code = "WITH ranked AS (\n  SELECT *, ROW_NUMBER() OVER (\n    PARTITION BY entity_id ORDER BY ts DESC, id DESC\n  ) AS rn FROM updates\n)\nSELECT COUNT(*) AS entities, COALESCE(SUM(value), 0) AS total\nFROM ranked WHERE rn = 1 AND state = 'active';"
    out.append(task("sql_latest_records", "SQL", "Последнее состояние", "Таблица updates содержит историю изменений. Что вернёт запрос? При равном ts более поздним считается больший id.",
        {"entities":len(live), "total":sum(r[3] for r in live)}, {"entities":"entities — активных объектов", "total":"total — сумма"},
        "Сначала выбирается последняя запись каждого объекта. Если именно она deleted, объект исключается целиком; нельзя заранее отфильтровать удалённые записи.", code=code,
        tables=[table("updates", ["id", "entity_id", "ts", "value", "state"], rows)]))
    if specialization == "backend":
        nums = [rng.randint(1, 15) for _ in range(6)]
        count = 2 if hard else 1
        tail = sum(x*x for x in nums[count:] if x % 2) if hard else sum(nums[count:])
        code = f"from itertools import islice\nvalues = {nums}\nit = iter(values)\nprefix = list(islice(it, {count}))\n"
        code += "tail = sum(x*x for x in it if x % 2)\n" if hard else "tail = sum(it)\n"
        code += "print({'prefix': sum(prefix), 'tail': tail})"
        out.append(task("python_iterator_state", "Python", "Состояние итератора", "Каковы значения prefix и tail в напечатанном словаре? Итератор используется последовательно; заново он не создаётся.",
            {"prefix":sum(nums[:count]), "tail":tail}, {"prefix":"prefix", "tail":"tail"},
            "islice уже извлёк начало из общего итератора. Следующий проход видит только оставшиеся элементы.", code=code))
        a, b, c = rng.sample(range(2, 30), 3)
        if hard:
            code = f"closed = []\ndef values():\n    try:\n        yield {a}\n        yield {b}\n    finally:\n        closed.append({c})\n\ng = values()\nfirst = next(g)\ng.close()\nprint({{'first': first, 'closed': sum(closed)}})"
            answers = {"first":a, "closed":c}
            explain = "next запускает генератор до первого yield. close выполняет finally; второй yield не потребляется."
        else:
            code = f"total = 0\nvisited = 0\nfor value in {list(map(str,[a,b])) + ['bad']}:\n    try:\n        total += int(value)\n    except ValueError:\n        pass\n    finally:\n        visited += 1\nprint({{'total': total, 'visited': visited}})"
            answers = {"total":a+b, "visited":3}
            explain = "Ошибочная строка не увеличивает total, но finally выполняется и для неё."
        out.append(task("python_exception_flow", "Python", "Завершение и finally", "Введите значения напечатанного словаря. Код выполняется в Python 3.", answers, {k:k for k in answers}, explain, code=code))
        amounts = {k:rng.randint(2, 20) for k in ("A", "B", "C")}
        keys = ["A", "B", "A", "C", "B", "C", "A"]
        stages = ["ok"]*len(keys) if not hard else ["before_commit", "after_commit", "ok", "before_commit", "ok", "ok", "ok"]
        rows = [[i+1,k,amounts[k],stages[i]] for i,k in enumerate(keys)]
        initial = rng.randint(200, 400)
        committed = {k for _,k,_,stage in rows if stage != "before_commit"}
        prompt = f"Начальный запас: {initial}. Запросы идут по seq. Ключ операции и уменьшение запаса фиксируются атомарно. Повтор уже зафиксированного key не меняет запас. before_commit — транзакция откатилась; after_commit — фиксация прошла, но ответ потерян; ok — обычный успех. Отрицательного запаса здесь не возникает. Каковы итоговый запас и число зафиксированных операций?"
        out.append(task("api_retry_ledger", "API", "Повтор после сбоя", prompt,
            {"stock":initial-sum(amounts[k] for k in committed), "writes":len(committed)},
            {"stock":"Итоговый запас", "writes":"Зафиксировано операций"},
            "Откат до commit не сохраняет ключ. Потеря ответа после commit не отменяет изменение; повтор с тем же ключом возвращает уже полученный результат.",
            tables=[table("requests", ["seq", "key", "amount", "stage"], rows)]))
        rows = [[i+1,rng.randint(1,5)] for i in range(9)]
        size = rng.choice([2,3,4])
        ordered = sorted(rows,key=lambda r:(r[1],r[0]),reverse=True)
        cursor = ordered[size-1]
        if hard:
            rows.append([99,6])
            page = ordered[size:2*size]
            code = f"GET /events?limit={size}&after_ts={cursor[1]}&after_id={cursor[0]}"
            prompt = "Порядок API: created_at DESC, id DESC. После первой страницы добавилась запись id=99; ниже уже новая таблица. Курсор — последняя запись первой страницы. Следующая страница берёт только пары (created_at, id), строго меньшие курсора. Назовите первый и последний id следующей страницы."
        else:
            page = ordered[:size]
            code = f"GET /events?limit={size}"
            prompt = "API сортирует по created_at DESC, затем id DESC. Назовите первый и последний id первой страницы указанного размера."
        out.append(task("api_cursor_page", "API", "Страница по курсору", prompt,
            {"first_id":page[0][0],"last_id":page[-1][0]}, {"first_id":"Первый id", "last_id":"Последний id"},
            "При равных временах порядок задаёт id. Строгое сравнение всей пары исключает сам курсор и более новые записи; сдвиг OFFSET не используется.", code=code,
            tables=[table("events",["id","created_at"],rows)]))
    else:
        rows = []
        for variant in ("A","B"):
            for segment in (("new","returning") if hard else ("all",)):
                visitors = rng.choice([100,200,400])
                conversion = rng.choice([5,10,20,30])
                rows.append([variant,segment,visitors,visitors*conversion//100])
        rate = {v:Decimal(sum(r[3] for r in rows if r[0]==v))/sum(r[2] for r in rows if r[0]==v) for v in ("A","B")}
        delta = (rate["B"]-rate["A"])*100
        second = (sum(Decimal(r[3])/r[2] for r in rows if r[0]=="B")-sum(Decimal(r[3])/r[2] for r in rows if r[0]=="A"))*50 if hard else (rate["B"]/rate["A"]-1)*100
        label = "Разница B−A при весе каждого сегмента 50%, п.п." if hard else "Относительный прирост B к A, %"
        prompt = "Посчитайте разницу общей конверсии B−A в процентных пунктах. " + ("Затем пересчитайте эту разницу при одинаковом составе: каждому сегменту дайте вес 50% внутри каждого варианта." if hard else "Затем посчитайте относительный прирост: (конверсия B / конверсия A − 1) × 100%.") + " Округлите до двух знаков; половины округляются от нуля. Эти числа сами по себе не доказывают причинный эффект."
        out.append(task("analytics_standardization", "Аналитика", "Состав аудитории и метрика", prompt,
            {"raw_delta":delta,"adjusted":second}, {"raw_delta":"Общая разница B−A, п.п.","adjusted":label},
            "Для общей конверсии складываются числители и знаменатели. Для фиксированного состава отдельно считаются конверсии сегментов с заданными весами; относительный прирост делится на базовую конверсию.",
            tables=[table("experiment",["variant","segment","visitors","conversions"],rows)], places={"raw_delta":2,"adjusted":2}))
        cutoff = 60
        rows = []
        for i in range(8):
            occurred = rng.randint(30,80)
            rows.append([i+1, f"f{i%4 if hard else i}", occurred, occurred+rng.randint(0,20), rng.randint(1,20), "target" if i==7 else "feature"])
        eligible = [r for r in rows if r[2]<=cutoff and r[3]<=cutoff and r[5]=="feature"]
        latest = {}
        for row in eligible:
            if row[1] not in latest or (row[2],row[0]) > (latest[row[1]][2],latest[row[1]][0]):latest[row[1]]=row
        visible = list(latest.values()) if hard else eligible
        prompt = "Прогноз строится на минуте 60. Можно использовать только feature, у которых occurred_at ≤ 60 и available_at ≤ 60; target исключается. " + ("Для каждого name возьмите последнюю доступную запись по (occurred_at, id). " if hard else "Имена в этой таблице уникальны. ") + "Сколько значений доступно и какова их сумма?"
        out.append(task("analytics_temporal_features", "Аналитика", "Что было известно на момент прогноза", prompt,
            {"features":len(visible),"total":sum(r[4] for r in visible)}, {"features":"Доступно значений", "total":"Сумма значений"},
            "Прошлая дата события не делает поздно доставленное значение доступным в прошлом. Отбор идёт по времени события, доступности и роли; затем выбирается последнее допустимое состояние.",
            tables=[table("features",["id","name","occurred_at","available_at","value","role"],rows)]))
        rows = [[i+1,rng.choice(["A","B"]),rng.randint(1,2),rng.randint(1,3),rng.choice(["upsert","upsert","delete"]),rng.randint(2,20)] for i in range(9)]
        state = {}
        for row in rows:
            key = (row[1],row[2])
            if not hard or key not in state or row[3]>state[key][3]:state[key]=row
        live = [r for r in state.values() if r[4]=="upsert"]
        prompt = "Обрабатывайте по seq. Ключ объекта — (source, event_id); источники независимы. " + ("Новое состояние заменяет старое только при строго большей version. Равные и меньшие версии игнорируются. " if hard else "Последняя доставленная запись заменяет старое состояние; version не участвует. ") + "delete удаляет объект, сохраняя его последнее состояние для обработки дальнейших записей. Назовите число живых объектов и сумму их value."
        out.append(task("quality_versioned_events", "Качество данных", "Порядок доставки и версия", prompt,
            {"live":len(live),"total":sum(r[5] for r in live)}, {"live":"Живых объектов", "total":"Сумма value"},
            "Состояние хранится по составному ключу. Tombstone не позволяет устаревшей записи восстановить удалённый объект при политике строгой версии.",
            tables=[table("deliveries",["seq","source","event_id","version","op","value"],rows)]))
        rows = [[i+1,rng.choice([1,2,3]) if hard else 2,rng.choice([0,2,10,-1,None,1.5]),rng.choice(["ms","s","min"])] for i in range(9)]
        accepted = [r for r in rows if type(r[2]) is int and r[2]>=0 and ((r[1]==1 and r[3]=="ms") or (r[1]==2 and r[3] in ("ms","s")))]
        total = sum(r[2]*(1000 if r[3]=="s" else 1) for r in accepted)
        prompt = "duration должна быть целым неотрицательным числом; NULL и дробные значения отклоняются. " + ("Версия 1 принимает только ms, версия 2 — ms или s; остальные версии отклоняются. " if hard else "Принимаются единицы ms или s; min отклоняется. ") + "Принятые значения переводятся в миллисекунды: 1 s = 1000 ms. Сколько строк принято и какова сумма миллисекунд?"
        out.append(task("quality_contract", "Качество данных", "Контракт и единицы измерения", prompt,
            {"accepted":len(accepted),"milliseconds":total}, {"accepted":"Принято строк", "milliseconds":"Сумма миллисекунд"},
            "Сначала проверяются тип, знак, версия и единицы. Перевод единиц выполняется только для прошедших контракт строк; ноль является допустимым значением.",
            tables=[table("measurements",["id","version","duration","unit"],rows)]))
    return out
