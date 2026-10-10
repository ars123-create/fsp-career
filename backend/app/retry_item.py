"""A task variant that separates specified transaction mistakes.

This construction checks three explicit wrong rules. It is not an estimate of
item difficulty, human discrimination, professional grade, or AI resistance.
"""
from copy import deepcopy
import random


def revise_retry_task(task, seed):
    if task["family"] != "api_retry_ledger":
        raise ValueError("Only the retry-ledger family is revised")
    rng = random.Random(seed ^ 0xF5F31008)
    keys = rng.sample(list("ABCDEFGH"), 4)
    amounts = dict(zip(keys, rng.sample(range(2, 21), 4)))
    initial = rng.randint(200, 400)
    # In-order operations per key; interleave different keys freely.
    stages = [["before_commit", "ok"], ["after_commit", "after_commit"],
              ["before_commit"], ["ok", "ok"]]
    pending = [list(block) for block in stages]
    rows = []
    while any(pending):
        index = rng.choice([i for i, block in enumerate(pending) if block])
        stage = pending[index].pop(0)
        key = keys[index]
        rows.append([len(rows) + 1, key, amounts[key], stage])
    committed = {}
    for _, key, amount, stage in rows:
        if stage != "before_commit":
            committed.setdefault(key, amount)
    revised = deepcopy(task)
    revised["prompt"] = (
        f"Начальный запас: {initial}. Запросы идут по seq. Ключ операции и уменьшение запаса "
        "фиксируются атомарно. Повтор уже зафиксированного key не меняет запас. "
        "before_commit — транзакция откатилась; after_commit — фиксация прошла, но ответ "
        "потерян; ok — обычный успех. Отрицательного запаса здесь не возникает. "
        "Каковы итоговый запас и число зафиксированных операций?"
    )
    revised["tables"] = [{"name": "requests", "columns": ["seq", "key", "amount", "stage"], "rows": rows}]
    revised["_correct"] = {"stock": str(initial - sum(committed.values())), "writes": str(len(committed))}
    revised["_explanation"] = (
        "Откат не сохраняет ключ и списание. Если последующей фиксации для этого ключа нет, "
        "операция не учитывается. Потеря ответа после commit сохраняет изменение даже без "
        "последующего ok. Повтор зафиксированного ключа не списывает запас ещё раз."
    )
    return revised
