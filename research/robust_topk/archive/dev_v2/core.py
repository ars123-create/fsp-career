"""Exact minimax SUM top-k regret under shared priorities and feature boxes.

Own formulation/adaptation of established robust selection, not a new solver.
See DESIGN.md and PROOFS.md for the scope of each mathematical guarantee.
The finite vertex list MUST describe the complete allowed weight polytope.
Numerical solutions use SciPy/HiGHS, not a symbolic proof checker.
"""
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

TOL = 1e-7


def _milp(*args, **kwargs):
    # HiGHS default MIP feasibility tolerance is 1e-6. Tighten it before
    # claiming a 1e-7 independently checked certificate. SciPy documents that
    # unknown wrapper options are passed to HiGHS; suppress only this warning.
    kwargs["options"] = {**kwargs.get("options", {}), "mip_feasibility_tolerance": 1e-9}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*mip_feasibility_tolerance.*", category=RuntimeWarning)
        return milp(*args, **kwargs)


@dataclass(frozen=True)
class WeightRegion:
    dimension: int
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()
    priorities: tuple[tuple[int, int], ...] = ()  # first >= second

    def vertices(self):
        d = self.dimension
        if not 1 <= d <= 5:
            raise ValueError("Weight dimension must be 1..5.")
        lo = np.array(self.lower or (0.,) * d, dtype=float)
        hi = np.array(self.upper or (1.,) * d, dtype=float)
        if (lo.shape != (d,) or hi.shape != (d,) or
                not np.isfinite([lo, hi]).all() or
                np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi)):
            raise ValueError("Invalid weight bounds.")
        A = list(-np.eye(d)) + list(np.eye(d))
        b = list(-lo) + list(hi)
        if len(self.priorities) > 20:
            raise ValueError("Too many priority constraints.")
        for a, c in set(self.priorities):
            if a == c or not (0 <= a < d and 0 <= c < d):
                raise ValueError("Invalid priority comparison.")
            row = np.zeros(d)
            row[c], row[a] = 1., -1.
            A.append(row)
            b.append(0.)
        A, b = np.asarray(A), np.asarray(b)
        vertices = []
        # A bounded polytope's vertices have d independent active equations,
        # including sum(w)=1. Also handles lower-dimensional polytopes.
        for active in combinations(range(len(b)), d - 1):
            equations = np.vstack([np.ones(d), A[list(active)]])
            if np.linalg.matrix_rank(equations, tol=1e-10) < d:
                continue
            w = np.linalg.solve(equations, np.r_[1., b[list(active)]])
            if np.any(A @ w > b + 1e-9):
                continue
            if not any(np.allclose(w, v, atol=1e-9, rtol=0) for v in vertices):
                vertices.append(w)
        if not vertices:
            raise ValueError("Weight constraints are inconsistent.")
        return np.array(sorted(vertices, key=lambda w: tuple(w)))

    def prefer(self, a, b):
        return WeightRegion(self.dimension, self.lower, self.upper,
                            self.priorities + ((a, b),))


def validated(lower, upper, vertices, k):
    lo, hi, vs = (np.asarray(a, dtype=float) for a in (lower, upper, vertices))
    if (lo.ndim != 2 or hi.shape != lo.shape or vs.ndim != 2 or
            vs.shape[1] != lo.shape[1] or len(vs) == 0 or lo.shape[1] == 0):
        raise ValueError("Incompatible feature/weight shapes.")
    if (not all(np.isfinite(a).all() for a in (lo, hi, vs)) or
            np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi) or
            np.any(vs < -1e-9) or not np.allclose(vs.sum(axis=1), 1., atol=1e-9, rtol=0)):
        raise ValueError("Features must be ordered in [0,1]; weights on simplex.")
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= len(lo):
        raise ValueError("k must be between zero and the candidate count.")
    return lo, hi, vs


def topk(scores, k):
    """Stable index tie break; result ordered by score, not subset index."""
    return tuple(int(i) for i in np.lexsort((np.arange(len(scores)), -np.asarray(scores)))[:k])


@dataclass(frozen=True)
class Witness:
    regret: float
    weights: tuple[float, ...]
    challenger: tuple[int, ...]


def _oracle(lo, hi, vs, selected):
    n, k = len(lo), len(selected)
    chosen = np.zeros(n, dtype=bool)
    chosen[list(selected)] = True
    scenario = np.where(chosen[:, None], lo, hi)
    best = Witness(-1., (), ())
    for w in vs:
        scores = scenario @ w
        challenger = topk(scores, k)
        regret = max(0., float(scores[list(challenger)].sum() - scores[chosen].sum()))
        if regret > best.regret:
            best = Witness(regret, tuple(float(x) for x in w), challenger)
    return best


def worst_case(lower, upper, vertices, selected):
    selected = tuple(selected)
    lo, hi, vs = validated(lower, upper, vertices, len(selected))
    if (len(set(selected)) != len(selected) or
            any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(lo) for i in selected)):
        raise ValueError("Selected indices must be distinct and in range.")
    return _oracle(lo, hi, vs, selected)


@dataclass(frozen=True)
class Selection:
    selected: tuple[int, ...]
    witness: Witness
    lower_bound: float
    absolute_gap: float
    status: str
    seconds: float
    vertices: int


def solve(lower, upper, vertices, k, *, time_limit=5., tie_break=True):
    """Return a feasible shortlist and its independently evaluated certificate.

    A time limit never becomes an optimality claim. The worst-case certificate
    for a returned set remains valid even if optimal selection was not found.
    """
    start = perf_counter()
    lo, hi, vs = validated(lower, upper, vertices, k)
    if not np.isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be finite and nonnegative.")
    n, m = len(lo), len(vs)
    nominal = (lo + hi) / 2 @ vs.mean(axis=0)
    fallback = topk(nominal, k)
    if k in (0, n):
        return Selection(fallback, _oracle(lo, hi, vs, fallback), 0., 0.,
                         "optimal", perf_counter() - start, m)
    if time_limit == 0:
        witness = _oracle(lo, hi, vs, fallback)
        return Selection(fallback, witness, 0., witness.regret, "fallback",
                         perf_counter() - start, m)
    # x[n], r[1], lambda[m], z[m*n]. See the top-k epigraph proof.
    r_index, lambda_start, z_start = n, n + 1, n + 1 + m
    size = z_start + m * n
    rows, cols, vals, lb, ub = [], [], [], [], []

    def constraint(entries, low, high=np.inf):
        row = len(lb)
        for col, value in entries:
            if value:
                rows.append(row)
                cols.append(col)
                vals.append(value)
        lb.append(low)
        ub.append(high)

    constraint(((i, 1.) for i in range(n)), k, k)
    for v, w in enumerate(vs):
        low, high = lo @ w, hi @ w
        lam = lambda_start + v
        for i in range(n):
            z = z_start + v * n + i
            constraint([(i, high[i] - low[i]), (lam, 1.), (z, 1.)], high[i])
        constraint([(r_index, 1.), (lam, -float(k))] +
                   [(z_start + v * n + i, -1.) for i in range(n)] +
                   [(i, low[i]) for i in range(n)], 0.)
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lb), size)).tocsc()
    constraints = LinearConstraint(matrix, np.array(lb), np.array(ub))
    integrality = np.r_[np.ones(n), np.zeros(size - n)]
    upper_bounds = np.ones(size)
    upper_bounds[r_index] = k
    objective = np.zeros(size)
    objective[r_index] = 1.
    result = _milp(objective, integrality=integrality,
                  bounds=Bounds(np.zeros(size), upper_bounds), constraints=constraints,
                  options={"time_limit": max(.001, time_limit - (perf_counter() - start)),
                           "mip_rel_gap": 0.})
    selected = fallback
    if result.x is not None:
        indices = np.flatnonzero(result.x[:n] > .5)
        if len(indices) == k and np.max(np.abs(result.x[:n] - np.round(result.x[:n]))) < TOL:
            selected = tuple(int(i) for i in indices)
    witness = _oracle(lo, hi, vs, selected)
    fallback_witness = _oracle(lo, hi, vs, fallback)
    if fallback_witness.regret < witness.regret:
        selected, witness = fallback, fallback_witness
    bound = getattr(result, "mip_dual_bound", None)
    bound = max(0., float(bound)) if bound is not None and np.isfinite(bound) else 0.
    if bound > witness.regret + TOL:
        raise RuntimeError("Solver bound contradicts independently checked shortlist.")
    bound = min(bound, witness.regret)
    is_optimal = result.status == 0 and witness.regret - bound <= TOL
    # Lexicographic secondary objective: observed midpoint utility, preserving
    # the primary optimum to numerical tolerance. It is not a second guarantee.
    remaining = time_limit - (perf_counter() - start)
    if tie_break and is_optimal and remaining > .01:
        secondary = np.zeros(size)
        secondary[:n] = -nominal
        upper_bounds[r_index] = min(k, witness.regret + 1e-9)
        refined = _milp(secondary, integrality=integrality,
                       bounds=Bounds(np.zeros(size), upper_bounds), constraints=constraints,
                       options={"time_limit": remaining, "mip_rel_gap": 0.})
        if refined.x is not None:
            indices = tuple(int(i) for i in np.flatnonzero(refined.x[:n] > .5))
            if len(indices) == k:
                other = _oracle(lo, hi, vs, indices)
                if other.regret <= witness.regret + TOL:
                    selected, witness = indices, other
    selected = tuple(sorted(selected, key=lambda i: (-nominal[i], i)))
    return Selection(selected, witness, bound, max(0., witness.regret - bound),
                     "optimal" if is_optimal else ("bounded" if result.x is not None else "fallback"),
                     perf_counter() - start, m)


def independent_envelope(lower, upper, vertices):
    """Conservative comparator discarding common employer weights across people."""
    lower, upper, vertices = validated(lower, upper, vertices, 0)
    return (lower @ vertices.T).min(axis=1)[:, None], (upper @ vertices.T).max(axis=1)[:, None]


def question_options(lower, upper, region, k, *, time_limit=1.):
    """All undecided pairwise skill-priority questions and BOTH branch results.

    Best among this finite question class only when every solve is optimal.
    Skipping or answering inconsistently does not shrink the uncertainty set.
    """
    vs = region.vertices()
    options = []
    for a, b in combinations(range(region.dimension), 2):
        delta = vs[:, a] - vs[:, b]
        if delta.min() >= -1e-9 or delta.max() <= 1e-9:
            continue  # the comparison is already implied
        branches = tuple(solve(lower, upper, region.prefer(i, j).vertices(), k,
                               time_limit=time_limit) for i, j in ((a, b), (b, a)))
        options.append({"pair": (a, b), "worst_after": max(s.witness.regret for s in branches),
                        "optimal": all(s.status == "optimal" for s in branches),
                        "branches": branches})
    return sorted(options, key=lambda item: (item["worst_after"], item["pair"]))
