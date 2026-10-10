"""Current practical bank; immutable v2 generation stays available separately."""
from app.practical_bank_v2 import (
    build_practical as build_practical_v2,
    correct, number, validate_values,
)
from app.retry_item import revise_retry_task

VERSION = "2026-10-v3"


def build_practical(specialization, grade, seed):
    tasks = build_practical_v2(specialization, grade, seed)
    if specialization == "backend" and grade == "Middle":
        tasks = [revise_retry_task(q, seed) if q["family"] == "api_retry_ledger" else q for q in tasks]
    return tasks
