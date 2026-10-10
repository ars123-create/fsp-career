from copy import deepcopy
from itertools import permutations

import pytest
from hypothesis import given, settings, strategies as st

from .legacy_practical_bank import build_practical
from .retry_item import revise_retry_task
from .evaluate import MODES, sql_reference, wrong_rule


def legacy(seed):
    return next(q for q in build_practical("backend", "Middle", seed) if q["family"] == "api_retry_ledger")


def test_hand_calculated_transaction_case():
    task = legacy(948500)
    task["prompt"] = "Начальный запас: 100."
    task["tables"][0]["rows"] = [
        [1, "A", 10, "before_commit"], [2, "B", 20, "after_commit"],
        [3, "D", 5, "ok"], [4, "A", 10, "ok"], [5, "C", 30, "before_commit"],
        [6, "B", 20, "after_commit"], [7, "D", 5, "ok"]]
    assert sql_reference(task) == {"stock": 65, "writes": 3}
    assert wrong_rule(task, "ignore_stage") == {"stock": 35, "writes": 4}
    assert wrong_rule(task, "lost_response_rolls_back") == {"stock": 85, "writes": 2}
    assert wrong_rule(task, "no_deduplication") == {"stock": 40, "writes": 5}


@settings(max_examples=20, derandomize=True, database=None)
@given(st.sampled_from(list(range(948500, 948520))))
def test_new_variants_reject_specified_rules_without_mutating_source(seed):
    old = legacy(seed)
    before = deepcopy(old)
    new = revise_retry_task(old, seed)
    expected = sql_reference(new)
    assert {k: int(v) for k, v in new["_correct"].items()} == expected
    assert all(wrong_rule(new, mode) != expected for mode in MODES)
    assert old == before
    assert new == revise_retry_task(old, seed)
    assert len(new["tables"][0]["rows"]) == 7
    assert {k: v for k, v in new.items() if k not in ("prompt", "tables", "_correct", "_explanation")} == {
        k: v for k, v in old.items() if k not in ("prompt", "tables", "_correct", "_explanation")}


def test_all_in_order_interleavings_retain_transaction_distinctions():
    task = legacy(948500)
    task["prompt"] = "Начальный запас: 100."
    blocks = [[("A", 10, "before_commit"), ("A", 10, "ok")],
              [("B", 20, "after_commit"), ("B", 20, "after_commit")],
              [("C", 30, "before_commit")], [("D", 5, "ok"), ("D", 5, "ok")]]
    orders = set(permutations((0, 0, 1, 1, 2, 3, 3)))
    assert len(orders) == 630
    for order in orders:
        positions = [0, 0, 0, 0]
        rows = []
        for seq, index in enumerate(order, 1):
            rows.append([seq, *blocks[index][positions[index]]])
            positions[index] += 1
        task["tables"][0]["rows"] = rows
        assert sql_reference(task) == {"stock": 65, "writes": 3}
        assert all(wrong_rule(task, mode) != sql_reference(task) for mode in MODES)


def test_wrong_family_rejected():
    task = legacy(948500)
    task["family"] = "unrelated"
    with pytest.raises(ValueError):
        revise_retry_task(task, 948500)
