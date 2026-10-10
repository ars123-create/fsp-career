"""Product adapter versus the unchanged research policy and analytic invariants."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
from app.mixed_engine.adapter import recommend
from research.frontier_2026.policies import Config, ParticleSession


@pytest.mark.parametrize("n,d", [(1, 1), (5, 3), (6, 2), (18, 3), (40, 2), (100, 3)])
@pytest.mark.parametrize("seed", [202610109100, 202610109101])
def test_matches_frozen_m6(n, d, seed):
    values = np.random.default_rng(seed).uniform(.05, 1, (n, d))
    source = ParticleSession(values, Config(m=6))
    history, answers = [], []
    for response in (0, 2, 1, None):
        expected, q = source.recommend(history)
        actual = recommend(values, answers)
        assert actual.selected == expected
        assert actual.question == (source.pairs[q] if q is not None else None)
        assert sum(actual.family_mass.values()) == pytest.approx(1)
        if q is None or response is None:
            break
        a, b = source.pairs[q]
        answers.append((values[a], values[b], ("left", "right", "skip")[response]))
        history.append((q, response))


def test_history_survives_removed_profile_and_skip_is_neutral():
    values = np.random.default_rng(202610109102).uniform(.05, 1, (18, 3))
    first = recommend(values)
    a, b = first.question
    skipped = recommend(values, [(values[a], values[b], "skip")])
    assert skipped.selected == first.selected
    np.testing.assert_array_equal(skipped.scores, first.scores)
    changed = np.delete(values, a, axis=0)
    result = recommend(changed, [(values[a], values[b], "left")])
    assert len(result.selected) == 5
    assert all(0 <= i < len(changed) for i in result.selected)


def test_response_orientation_and_dominance():
    values = np.array([[1., 0], [0, 1], [.8, .8], [.1, .1], [.7, .7], [.6, .6], [.5, .5]])
    a = recommend(values, [(values[0], values[1], "left")])
    b = recommend(values, [(values[1], values[0], "right")])
    assert a.selected == b.selected
    np.testing.assert_array_equal(a.scores, b.scores)
    for i in a.selected:
        for j in set(range(len(values)))-set(a.selected):
            assert not (np.all(values[j] >= values[i]) and np.any(values[j] > values[i]))


def test_invalid_or_repeated_comparisons_rejected():
    values = np.array([[1., 0], [0, 1], [.5, .5]])
    with pytest.raises(ValueError, match="repeated"):
        recommend(values, [(values[0], values[1], "left"), (values[1], values[0], "skip")])
    with pytest.raises(ValueError, match="trade-off"):
        recommend(values, [([1., 1], [.1, .1], "left")])
