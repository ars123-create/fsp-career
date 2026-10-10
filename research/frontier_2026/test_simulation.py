import numpy as np
import pytest

from .simulation import FAMILIES, evaluate, make_world, response


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_synthetic_world_is_normalized_monotone_and_reproducible(family, d):
    a = make_world(202610080700, family, n=30, d=d)
    b = make_world(202610080700, family, n=30, d=d)
    assert np.array_equal(a.values, b.values) and np.array_equal(a.truth, b.truth)
    assert np.isfinite(a.truth).all() and np.min(a.truth) >= -1e-12 and np.max(a.truth) <= 1+1e-12
    for i in range(len(a.values)):
        for j in range(len(a.values)):
            if np.all(a.values[i] >= a.values[j]):
                assert a.truth[i] >= a.truth[j]-1e-12
    oracle = tuple(map(int, np.argsort(-a.truth, kind="stable")[:5]))
    assert evaluate(a, oracle)['regret'] == 0


def test_respondent_probabilities_and_common_draws_are_fixed():
    a = make_world(202610080701, 'query_skips')
    for slot in range(3):
        first = response(a, (0, 1), slot)
        assert first == response(a, (0, 1), slot)
        answer, p, s = first
        assert answer in (0, 1, 2) and 0 <= p <= 1 and 0 <= s <= 1
        assert response(a, (1, 0), slot)[1] == pytest.approx(1-p)
    with pytest.raises(ValueError):
        response(a, (0, 1), 3)


def test_evaluator_cannot_accept_invalid_shortlist():
    a = make_world(202610080702, 'linear')
    for bad in [(0,)*5, (0, 1), (0, 1, 2, 3, 18)]:
        with pytest.raises(ValueError):
            evaluate(a, bad)


def test_overlap_accepts_any_utility_tied_optimal_slate():
    from dataclasses import replace
    world = make_world(71, 'linear', n=10)
    world = replace(world, values=np.full((10, 3), .5), truth=np.full(10, .5))
    first, last = evaluate(world, (0, 1, 2, 3, 4)), evaluate(world, (5, 6, 7, 8, 9))
    assert first['regret'] == last['regret'] == 0
    assert first['oracle_overlap_tie_aware'] == last['oracle_overlap_tie_aware'] == 1
    assert first['oracle_overlap_index_tiebreak'] == 1
    assert last['oracle_overlap_index_tiebreak'] == 0


def test_response_probability_partition_and_scale_transforms():
    from .simulation import VARIANTS, response_probabilities
    base = make_world(71, 'linear', n=10)
    for variant in VARIANTS:
        world = make_world(71, 'linear', n=10, variant=variant)
        p = response_probabilities(world, (0, 1))
        np.testing.assert_allclose(p.sum(), 1)
        assert (p >= 0).all() and (p <= 1).all()
        if variant in {'utility_square', 'utility_sqrt'}:
            assert np.array_equal(np.argsort(base.truth), np.argsort(world.truth))
        if variant == 'measurement_error':
            assert np.array_equal(base.truth, world.truth)
            assert not np.array_equal(base.values, world.values)


def test_query_coupling_is_history_independent_and_orientation_consistent():
    world = make_world(9910, 'linear', variant='query_coupling')
    left = [response(world, (0, 1), slot)[0] for slot in range(3)]
    right = [response(world, (1, 0), slot)[0] for slot in range(3)]
    assert len(set(left)) == len(set(right)) == 1
    assert (left[0] == right[0] == 2) or left[0]+right[0] == 1
