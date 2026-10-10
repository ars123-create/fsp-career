"""Explicit finite priors; these are modelling hypotheses, not employer data."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product

import numpy as np
from scipy.stats import qmc

from research.preference_topk.core import simplex_particles
from .finite import tradeoff_pairs

PRIOR_SEED = 2026100802


@dataclass(frozen=True)
class UtilityPrior:
    utilities: np.ndarray
    probabilities: np.ndarray
    families: tuple[str, ...]
    name: str


def _values(values):
    x = np.asarray(values, dtype=float)
    tradeoff_pairs(x)  # common bounded feature contract; no labels or identity
    return x


def standard_prior(values, *, kind="mixed", m=4, seed=PRIOR_SEED):
    """Symmetric simplex quadrature and an equally weighted mixture of families.

    Six families use the same symmetric weight nodes: linear, geometric,
    generalized means p=-2 and p=2, diminishing-return exponential, and a
    minimum-plus-linear utility. Every function maps all-zero/all-one anchors
    to 0/1 (geometric uses a numerical floor at zero). Mixture weights are
    declared, not learned from test outcomes. This is a finite approximation.
    """
    x = _values(values)
    w = simplex_particles(x.shape[1], m=m, seed=seed)
    linear = w @ x.T
    safe = np.maximum(x, 1e-12)
    families = {"linear": linear}
    if kind == "mixed":
        families.update({
            "geometric": np.exp(w @ np.log(safe).T),
            "power_minus2": (w @ (safe**-2).T)**-.5,
            "power_plus2": np.sqrt(w @ (x*x).T),
            "saturation": w @ ((1-np.exp(-3*x))/(1-np.exp(-3))).T,
            "bottleneck": .4*linear+.6*np.min(x, axis=1)[None, :],
        })
    elif kind != "linear":
        raise ValueError("kind must be linear or mixed")
    u = np.concatenate(list(families.values()), axis=0)
    tags = tuple(name for name, block in families.items() for _ in block)
    p = np.full(len(u), 1/len(u))
    return UtilityPrior(u, p, tags, kind)


def choquet_prior(values, *, m=6, seed=PRIOR_SEED):
    """P05: nonnegative mixtures of xi, min(xi,xj), and max(xi,xj).

    Since max(a,b)=a+b-min(a,b), these are normalized 2-additive Choquet
    utilities with positive AND and OR components. This parameterization is
    sufficient for monotonicity but does not claim a uniform prior over all
    admissible capacities. Feature permutations enforce a symmetric prior.
    """
    x = _values(values)
    d = x.shape[1]
    pairs = tuple(combinations(range(d), 2))
    dim = d+2*len(pairs)
    base = qmc.Sobol(dim, scramble=True, rng=seed).random_base2(m)
    weights = -np.log(np.maximum(base, 1e-15))
    weights /= weights.sum(axis=1, keepdims=True)
    output = []
    for perm in permutations(range(d)):
        xp = x[:, perm]
        features = np.column_stack([xp[:, j] for j in range(d)] +
                                   [np.minimum(xp[:, a], xp[:, b]) for a, b in pairs] +
                                   [np.maximum(xp[:, a], xp[:, b]) for a, b in pairs])
        output.append(weights @ features.T)
    u = np.concatenate(output)
    return UtilityPrior(u, np.full(len(u), 1/len(u)), ("choquet2",)*len(u), "choquet2")


@lru_cache(maxsize=24)
def lattice_knots(d, count=64, seed=PRIOR_SEED):
    """Random linear extensions of the 3^d lattice plus ordered knot heights.

    This samples an explicit monotone finite prior. It is not fitted TensorFlow
    Lattice and not uniform over the monotone polytope. All coordinate
    permutations are included later. Caching depends only on declared settings.
    """
    if not 1 <= d <= 3 or not isinstance(count, int) or count < 1:
        raise ValueError("invalid lattice dimensions/count")
    coordinates = tuple(product(range(3), repeat=d))
    lookup = {node: i for i, node in enumerate(coordinates)}
    predecessors = []
    for node in coordinates:
        pred = set()
        for axis, level in enumerate(node):
            if level:
                before = list(node)
                before[axis] -= 1
                pred.add(lookup[tuple(before)])
        predecessors.append(pred)
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(count):
        remaining = set(range(len(coordinates)))
        done = set()
        ordering = []
        while remaining:
            ready = sorted(i for i in remaining if predecessors[i] <= done)
            chosen = int(rng.choice(ready))
            ordering.append(chosen)
            remaining.remove(chosen)
            done.add(chosen)
        heights = np.r_[0., np.sort(rng.uniform(0, 1, len(coordinates)-2)), 1.]
        row = np.empty(len(coordinates))
        row[ordering] = heights
        output.append(row)
    result = np.array(output)
    result.setflags(write=False)
    return result


def lattice_prior(values, *, count=64, seed=PRIOR_SEED):
    """P04: multilinear interpolation of monotone lattice utility particles."""
    x = _values(values)
    d = x.shape[1]
    knots = lattice_knots(d, count, seed)
    output = []
    radix = 3**np.arange(d-1, -1, -1)
    for perm in permutations(range(d)):
        scaled = x[:, perm]*2
        lower = np.minimum(np.floor(scaled).astype(int), 1)
        fraction = scaled-lower
        utilities = np.zeros((len(knots), len(x)))
        for bits in product((0, 1), repeat=d):
            bits = np.array(bits)
            index = (lower+bits) @ radix
            weight = np.prod(np.where(bits, fraction, 1-fraction), axis=1)
            utilities += knots[:, index]*weight[None, :]
        output.append(utilities)
    u = np.concatenate(output)
    return UtilityPrior(u, np.full(len(u), 1/len(u)), ("lattice",)*len(u), "lattice")
