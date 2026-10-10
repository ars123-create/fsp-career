"""Harness-only checks, with no algorithm benchmark or frozen evidence discovery."""
from .audit import CORE, EXTRAS, canonical_pairs, fixtures, grid, safe_json

import numpy as np


def test_predeclared_coverage_and_no_duplicate_cells():
    f = fixtures()
    assert len(f) == 9
    assert {v["n"] for v in f.values()} == {1, 5, 6, 50, 100}
    assert {v["d"] for v in f.values()} == {1, 2, 3}
    assert len(grid()) == len(set(grid())) == 80
    assert {m for m, _ in grid()} == set(CORE+EXTRAS)
    assert all(name in f for _, name in grid())


def test_fixture_independence_and_canonical_oracle():
    f = fixtures()
    assert f == fixtures()
    assert len(canonical_pairs(f["antichain_100x3"]["values"])) == 4950
    for name in ("singleton_1x1", "all_equal_5x3", "chain_50x1", "chain_50x3"):
        assert canonical_pairs(f[name]["values"]) == []
    assert len(canonical_pairs(f["mostly_chain_100x2"]["values"])) == 10
    assert len(canonical_pairs([[0, 1], [1, 0], [0, 1], [0, 0]])) == 1


def test_json_preserves_nonfinite_diagnostics_without_invalid_json():
    assert safe_json({"x": np.array([1., np.inf]), "y": np.int64(2)}) == {"x": [1., "inf"], "y": 2}
