"""Finite-belief decision-EVOI for SUM top-k, using SciPy QMC and expit.

Known Bayesian elicitation, not a novel learning algorithm. See DESIGN/PROOFS.
All values are conditional on a specified utility and response model. A computed
gain is neither a hiring-success probability nor a real-person quality bound.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations

import numpy as np
from scipy.special import expit, xlogy
from scipy.stats import qmc


def topk(scores, k):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(scores):
        raise ValueError("k must be between 1 and the number of candidates")
    return tuple(int(i) for i in np.argsort(-scores, kind="stable")[:k])


def topk_sum(values, k):
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise ValueError("values must have a finite final candidate axis")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= values.shape[-1]:
        raise ValueError("invalid k")
    return np.partition(values, values.shape[-1] - k, axis=-1)[..., -k:].sum(axis=-1)


@lru_cache(maxsize=16)
def simplex_particles(d=3, m=9, seed=202610072):
    """Uniform simplex quadrature with exact permutation symmetry (d <= 3).

    The base Sobol design has 2**m nodes and is never skipped or thinned. Its
    symmetrized simplex transform has factorial(d)*2**m nodes. No Sobol balance
    guarantee is asserted for that transformed array or a conditioned subset.
    """
    if not isinstance(d, int) or not 1 <= d <= 3:
        raise ValueError("this bounded research adapter supports 1..3 skills")
    if not isinstance(m, int) or not 1 <= m <= 14:
        raise ValueError("m must be an integer in 1..14")
    if d == 1:
        result = np.ones((1, 1))
    else:
        base = qmc.Sobol(d=d-1, scramble=True, rng=seed).random_base2(m)
        cuts = np.c_[np.zeros(len(base)), np.sort(base, axis=1), np.ones(len(base))]
        gaps = np.diff(cuts, axis=1)
        result = np.concatenate([gaps[:, p] for p in permutations(range(d))])
    result.setflags(write=False)
    return result


def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return -xlogy(p, p) - xlogy(1-p, 1-p)


def response_probability(difference, temperature=.04, lapse=.02):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(lapse) or not 0 <= lapse <= 1:
        raise ValueError("lapse must be in [0, 1]")
    difference = np.asarray(difference, dtype=float)
    if not np.isfinite(difference).all():
        raise ValueError("utility differences must be finite")
    return lapse/2 + (1-lapse)*expit(difference/temperature)


@dataclass(frozen=True)
class Belief:
    """Utility hypotheses (particles x candidates) and posterior probabilities."""
    utilities: np.ndarray
    probabilities: np.ndarray | None = None

    def __post_init__(self):
        u = np.array(self.utilities, dtype=float, copy=True)
        if u.ndim != 2 or min(u.shape) < 1 or not np.isfinite(u).all():
            raise ValueError("nonempty finite utility matrix required")
        if np.any((u < 0) | (u > 1)):
            raise ValueError("normalized utilities must be in [0, 1]")
        p = (np.full(len(u), 1/len(u)) if self.probabilities is None
             else np.array(self.probabilities, dtype=float, copy=True))
        if p.shape != (len(u),) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("invalid probability vector")
        if not np.isclose(p.sum(), 1., atol=1e-12, rtol=1e-12):
            raise ValueError("probabilities must sum to one")
        p /= p.sum()
        u.setflags(write=False)
        p.setflags(write=False)
        object.__setattr__(self, "utilities", u)
        object.__setattr__(self, "probabilities", p)

    @classmethod
    def linear(cls, values, *, m=9, seed=202610072):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
            raise ValueError("nonempty finite candidate x feature matrix required")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("observed features must be normalized to [0, 1]")
        return cls(simplex_particles(values.shape[1], m, seed) @ values.T)

    @property
    def scores(self):
        return self.probabilities @ self.utilities

    def ranking(self, k):
        return topk(self.scores, k)

    def expected_regret(self, k):
        # Conditional on these finite hypotheses, not a real-world certificate.
        return float(self.probabilities @ topk_sum(self.utilities, k)
                     - topk_sum(self.scores, k))

    def likelihood(self, pair, *, temperature=.04, lapse=.02):
        a, b = _pair(pair, self.utilities.shape[1])
        return response_probability(self.utilities[:, a] - self.utilities[:, b],
                                    temperature, lapse)

    def observe(self, pair, choice, *, temperature=.04, lapse=.02):
        """choice=0/1 selects the left/right option; None explicitly abstains."""
        likelihood = self.likelihood(pair, temperature=temperature, lapse=lapse)
        if choice is None:
            return self
        if type(choice) not in (int, np.int64, np.int32) or choice not in (0, 1):
            raise ValueError("choice must be 0, 1, or None")
        if choice == 1:
            likelihood = 1-likelihood
        unnormalized = self.probabilities * likelihood
        if unnormalized.sum() <= 0:
            raise ValueError("answer has zero probability under this model")
        return Belief(self.utilities, unnormalized / unnormalized.sum())


def _pair(pair, n):
    if len(pair) != 2 or any(not isinstance(i, (int, np.integer)) for i in pair):
        raise ValueError("pair must contain two candidate indices")
    a, b = pair
    if a == b or not (0 <= a < n and 0 <= b < n):
        raise ValueError("pair indices must be distinct and in range")
    return int(a), int(b)


@dataclass(frozen=True)
class Question:
    pair: tuple[int, int]
    expected_gain: float
    information_gain: float
    answer_entropy: float
    qeubo: float
    probability_left: float


def questions(belief, k, *, asked=(), temperature=.04, lapse=.02):
    """Enumerate the common finite pool; each score is for ONE next question."""
    n = belief.utilities.shape[1]
    prior_value = float(topk_sum(belief.scores, k))
    excluded = {tuple(sorted(_pair(pair, n))) for pair in asked}
    pairs = [p for p in combinations(range(n), 2) if p not in excluded]
    if not pairs:
        return []
    output = []
    # Bound intermediate storage; never drop pairs based on a policy's score.
    for start in range(0, len(pairs), 128):
        block = pairs[start:start+128]
        indices = np.asarray(block)
        left = belief.utilities[:, indices[:, 0]].T
        right = belief.utilities[:, indices[:, 1]].T
        likelihood = response_probability(left-right, temperature, lapse)
        unnormalized = likelihood * belief.probabilities
        probability_left = unnormalized.sum(axis=1)
        first = unnormalized @ belief.utilities
        second = belief.scores - first
        gains = topk_sum(first, k) + topk_sum(second, k) - prior_value
        entropy = binary_entropy(np.clip(probability_left, 0, 1))
        info = entropy - binary_entropy(likelihood) @ belief.probabilities
        qeubo = np.maximum(left, right) @ belief.probabilities
        if np.any(gains < -1e-10) or np.any(info < -1e-10):
            raise ArithmeticError("negative information value beyond roundoff")
        output.extend(Question(pair, float(max(gains[i], 0)), float(max(info[i], 0)),
                               float(entropy[i]), float(qeubo[i]), float(probability_left[i]))
                      for i, pair in enumerate(block))
    return output


def select_question(options, policy, rng=None):
    if not options:
        return None
    if policy == "random":
        if rng is None:
            raise ValueError("random baseline requires an explicit generator")
        return options[int(rng.integers(len(options)))]
    attribute = {"decision_evoi": "expected_gain", "bald": "information_gain",
                 "uncertainty": "answer_entropy", "qeubo": "qeubo"}.get(policy)
    if attribute is None:
        raise ValueError("unknown query policy")
    return max(options, key=lambda q: getattr(q, attribute))
