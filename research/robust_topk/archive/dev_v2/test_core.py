from itertools import combinations, product

import numpy as np
import pytest

from .core import (WeightRegion, independent_envelope, question_options,
                   solve, topk, worst_case)


def test_oracle_against_all_feature_corners_and_all_shortlists():
    rng = np.random.default_rng(811)
    vs = WeightRegion(2, lower=(.1, .2)).vertices()
    for _ in range(10):
        lo = rng.uniform(.1, .65, (3, 2))
        hi = lo + rng.uniform(0., .3, (3, 2))
        for chosen in combinations(range(3), 2):
            brute = 0.
            for corner in product((0, 1), repeat=6):
                values = np.where(np.array(corner).reshape(3, 2), hi, lo)
                for w in vs:
                    scores = values @ w
                    best = max(sum(scores[i] for i in other) for other in combinations(range(3), 2))
                    brute = max(brute, best - sum(scores[i] for i in chosen))
            actual = worst_case(lo, hi, vs, chosen)
            assert actual.regret == pytest.approx(brute, abs=1e-9)


@pytest.mark.parametrize("seed", range(30))
def test_solver_matches_exhaustive_subset_optimum(seed):
    rng = np.random.default_rng(9200 + seed)
    lo = rng.uniform(.05, .7, (7, 3))
    hi = np.minimum(1., lo + rng.uniform(0., .3, lo.shape))
    region = WeightRegion(3, lower=(.05, .1, .1), priorities=((0, 1),) if seed % 2 else ())
    vertices = region.vertices()
    brute = min(worst_case(lo, hi, vertices, s).regret for s in combinations(range(7), 3))
    actual = solve(lo, hi, vertices, 3)
    assert actual.status == "optimal"
    assert actual.witness.regret == pytest.approx(brute, abs=1e-7)
    assert actual.absolute_gap < 1e-7


def test_strict_improvement_and_average_counterexample():
    values = np.array([[.95, .5], [.5, .95], [.7, .7]])
    vs = WeightRegion(2).vertices()
    robust = solve(values, values, vs, 1)
    nominal = topk(values.mean(axis=1), 1)
    assert robust.selected == (2,)
    assert robust.witness.regret == pytest.approx(.25)
    assert worst_case(values, values, vs, nominal).regret == pytest.approx(.45)
    # Robustness is NOT pointwise or average superiority.
    assert values[robust.selected[0]].mean() < values[nominal[0]].mean()


def test_shared_weights_tighter_than_independent_envelope():
    values = np.array([[.4, .8], [.5, .9]])
    vs = WeightRegion(2).vertices()
    lo, hi = independent_envelope(values, values, vs)
    assert worst_case(values, values, vs, (1,)).regret == pytest.approx(0.)
    assert worst_case(lo, hi, [[1.]], (1,)).regret == pytest.approx(.3)


def test_point_region_recovers_topk_and_degenerate_regions():
    values = np.random.default_rng(55).uniform(size=(8, 3))
    vs = WeightRegion(3, lower=(.2, .3, .5), upper=(.2, .3, .5)).vertices()
    assert len(vs) == 1
    result = solve(values, values, vs, 4)
    assert set(result.selected) == set(topk(values @ vs[0], 4))
    assert result.witness.regret == pytest.approx(0.)
    assert np.allclose(WeightRegion(3, priorities=((0, 1), (1, 2), (2, 0))).vertices(), [[1/3]*3])
    assert np.allclose(WeightRegion(1).vertices(), [[1.]])


def test_sampled_world_certificate_and_nested_region_monotonicity():
    rng = np.random.default_rng(20261007)
    lo = rng.uniform(.05, .65, (9, 3))
    hi = lo + rng.uniform(0., .3, lo.shape)
    region = WeightRegion(3)
    result = solve(lo, hi, region.vertices(), 3)
    for _ in range(500):
        scores = rng.uniform(lo, hi) @ rng.dirichlet(np.ones(3))
        regret = scores[list(topk(scores, 3))].sum() - scores[list(result.selected)].sum()
        assert regret <= result.witness.regret + 1e-9
    options = question_options(lo, hi, region, 3)
    assert len(options) == 3
    assert all(item["optimal"] for item in options)
    assert all(item["worst_after"] <= result.witness.regret + 1e-7 for item in options)
    narrow = solve((lo + hi) / 2, hi, region.prefer(0, 1).vertices(), 3)
    assert narrow.witness.regret <= result.witness.regret + 1e-7


def test_zero_budget_fallback_is_explicit_and_still_certified():
    values = np.array([[.95, .5], [.5, .95], [.7, .7]])
    result = solve(values, values, WeightRegion(2).vertices(), 1, time_limit=0)
    assert result.status == "fallback"
    assert result.witness.regret == pytest.approx(.45)
    assert result.absolute_gap == pytest.approx(.45)
    assert result.lower_bound == 0


def test_invalid_and_boundary_inputs():
    with pytest.raises(ValueError):
        WeightRegion(2, lower=(.7, .7)).vertices()
    with pytest.raises(ValueError):
        solve([[.8]], [[.2]], [[1.]], 1)
    with pytest.raises(ValueError):
        solve([[np.nan]], [[1.]], [[1.]], 1)
    with pytest.raises(ValueError):
        solve([[.2]], [[1.]], [[.5]], 1)
    with pytest.raises(ValueError):
        worst_case([[.2], [.3]], [[.5], [.5]], [[1.]], (0, 0))
    values = [[.3, .5], [.4, .6]]
    vs = WeightRegion(2).vertices()
    for k in (0, 2):
        assert solve(values, values, vs, k).witness.regret == 0
    assert question_options(values, values, WeightRegion(2, priorities=((0, 1),)), 1) == []
