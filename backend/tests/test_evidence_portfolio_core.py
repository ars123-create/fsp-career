from copy import deepcopy
from itertools import product

import pytest

from app.evidence_portfolio import refresh_portfolio, start_portfolio, summarize

CONTEXT = {"specialization": "backend", "grade": "Middle", "rubric_version": "2026-10-v2"}


def rows(values=(100, 100, 100, 0), timestamp=100):
    return [{"skill": "SQL", "family": f"f{i}", "score": value, "checked_at": timestamp}
            for i, value in enumerate(values)]


def test_manual_example_preserves_unmeasured_families_and_oldest_date():
    base = start_portfolio(CONTEXT, rows(), "full")
    before = deepcopy(base)
    updated = refresh_portfolio(base, CONTEXT, [rows((0,), 200)[0]], "short")
    result = summarize(updated)["SQL"]
    assert result == {"score": 50, "families": 4, "checked_at": 100, "latest_checked_at": 200,
                      "source": "family-latest-v1", "rubric_version": "2026-10-v2"}
    assert base == before
    for family in ("f1", "f2", "f3"):
        assert updated["families"]["SQL"][family] == before["families"]["SQL"][family]


def test_exhaustive_four_family_states_match_independent_arithmetic():
    checked = 0
    for bits in product((0, 1), repeat=4):
        base = start_portfolio(CONTEXT, rows([100*b for b in bits]), "full")
        for index in range(4):
            for replacement in (0, 1):
                observation = rows([100*b for b in bits], 200)[index]
                observation["score"] = replacement*100
                result = summarize(refresh_portfolio(base, CONTEXT, [observation], "short"))["SQL"]
                expected = list(bits);expected[index] = replacement
                assert result["score"] == 25*sum(expected)
                assert abs(result["score"]-25*sum(bits)) <= 25
                assert result["families"] == 4
                checked += 1
    assert checked == 128


def test_repetition_does_not_increase_coverage_or_count_attempts_as_families():
    result = start_portfolio(CONTEXT, rows(), "full")
    for index in range(10):
        observation = rows(timestamp=200+index)[3]
        observation["score"] = 100 if index % 2 else 0
        result = refresh_portfolio(result, CONTEXT, [observation], f"short-{index}")
        assert summarize(result)["SQL"]["families"] == 4
        assert summarize(result)["SQL"]["score"] == (100 if index % 2 else 75)
    # An old attempt ID cannot roll back a newer observation.
    assert refresh_portfolio(result, CONTEXT, [rows(timestamp=200)[3]], "short-0") == result


@pytest.mark.parametrize("field,value", [("grade", "Junior"), ("specialization", "data"), ("rubric_version", "2026-10-v1")])
def test_contexts_never_mix(field, value):
    base = start_portfolio(CONTEXT, rows(), "full")
    with pytest.raises(ValueError):
        refresh_portfolio(base, CONTEXT | {field:value}, rows(timestamp=200), "short")


@pytest.mark.parametrize("change", [{"family":"new"}, {"skill":"API"}, {"checked_at":99}, {"score":True},
                                     {"score":float("nan")}, {"score":50}, {"score":"100"}, {"checked_at":1.5}])
def test_invalid_refresh_is_atomic(change):
    base = start_portfolio(CONTEXT, rows(), "full"); before = deepcopy(base)
    valid = rows(timestamp=200)[0]
    invalid = rows(timestamp=200)[1] | change
    with pytest.raises(ValueError):
        refresh_portfolio(base, CONTEXT, [valid, invalid], "short")
    assert base == before


def test_empty_and_duplicate_observations_are_rejected():
    with pytest.raises(ValueError):
        start_portfolio(CONTEXT, [], "full")
    with pytest.raises(ValueError):
        start_portfolio(CONTEXT, [rows()[0], rows()[0]], "full")


def test_new_full_assessment_resets_basis_and_does_not_merge_contexts():
    old = start_portfolio(CONTEXT, rows(), "full")
    new = start_portfolio(CONTEXT | {"grade":"Junior"}, rows((100,100), 300), "new")
    assert summarize(new)["SQL"]["families"] == 2
    assert new["applied_attempts"] == ["new"]
    assert old["context"]["grade"] == "Middle"
