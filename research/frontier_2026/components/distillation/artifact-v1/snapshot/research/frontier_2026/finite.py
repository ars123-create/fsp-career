"""Finite Bayesian decision reference with exact categorical look-ahead.

The equations are standard finite-horizon Bayesian decision theory. This narrow
implementation supports a finite hypothesis table, up to three questions, an
explicit skip likelihood, no repeat questions, and a SUM top-k terminal reward.
No claim about real employers or verified Python execution follows from the
separate Lean proofs. See formal/ for the mathematical model and its assumptions.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.special import logsumexp, xlogy

from research.preference_topk.core import response_probability, topk_sum


def _integer(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def tradeoff_pairs(values):
    """One canonical unordered pair per distinct incomparable skill-vector pair."""
    x = np.asarray(values, dtype=float)
    if (x.ndim != 2 or not 1 <= x.shape[0] <= 100 or not 1 <= x.shape[1] <= 3
            or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))):
        raise ValueError("expected 1..100 profiles, 1..3 finite skills in [0,1]")
    representatives = {}
    for i, row in enumerate(x):
        representatives.setdefault(tuple(row), i)
    canonical = sorted(representatives)
    return tuple((representatives[a], representatives[b])
                 for a, b in combinations(canonical, 2)
                 if not (all(ai >= bi for ai, bi in zip(a, b))
                         or all(bi >= ai for ai, bi in zip(a, b))))


@dataclass(frozen=True)
class FiniteModel:
    """Utilities H×N, categorical likelihood Q×R×H, and normalized prior H.

    R can include an explicit abstention outcome; the FSP adapter uses R=3.
    Utilities are latent scores, not percentages. One-step values use homogeneous
    unnormalized masses; deeper planning rescales conditional masses at each
    node for stability and weights each continuation by its branch probability.
    """
    utilities: np.ndarray
    likelihood: np.ndarray
    prior: np.ndarray
    k: int

    def __post_init__(self):
        u = np.array(self.utilities, dtype=float, copy=True)
        ell = np.array(self.likelihood, dtype=float, copy=True)
        p = np.array(self.prior, dtype=float, copy=True)
        if u.ndim != 2 or min(u.shape) < 1 or not np.isfinite(u).all():
            raise ValueError("utilities must be a nonempty finite H by N matrix")
        k = _integer(self.k, "k", 1, u.shape[1])
        if (ell.ndim != 3 or ell.shape[1] < 2 or ell.shape[2] != len(u)
                or not np.isfinite(ell).all() or np.any((ell < 0) | (ell > 1))
                or not np.allclose(ell.sum(axis=1), 1, rtol=0, atol=8e-15)):
            raise ValueError("likelihood must be Q by R by H with outcome sums one")
        if (p.shape != (len(u),) or not np.isfinite(p).all() or np.any(p < 0)
                or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("prior must be nonnegative and sum to one")
        p /= p.sum()
        for name, data in (("utilities", u), ("likelihood", ell), ("prior", p)):
            data.setflags(write=False)
            object.__setattr__(self, name, data)
        object.__setattr__(self, "k", k)

    @classmethod
    def pairwise(cls, utilities, pairs, k=5, *, prior=None,
                 temperature=.04, lapse=.02, skip_probability=0.):
        u = np.asarray(utilities, dtype=float)
        if u.ndim != 2 or min(u.shape) < 1:
            raise ValueError("utilities must be H by N")
        checked = []
        seen = set()
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("a question needs two indices")
            a, b = (_integer(i, "candidate", 0, u.shape[1]-1) for i in pair)
            key = tuple(sorted((a, b)))
            if a == b or key in seen:
                raise ValueError("repeated or self comparison")
            seen.add(key)
            checked.append((a, b))
        skip = np.broadcast_to(np.asarray(skip_probability, dtype=float), (len(checked),))
        if not np.isfinite(skip).all() or np.any((skip < 0) | (skip > 1)):
            raise ValueError("skip probabilities must be in [0,1]")
        # This convenience adapter takes scalar response parameters.
        # A latent-noise extension builds its likelihood table explicitly.
        difference = np.asarray([u[:, a]-u[:, b] for a, b in checked])
        if not len(checked):
            difference = np.empty((0, len(u)))
        left = response_probability(difference, temperature, lapse)
        ell = np.stack(((1-skip[:, None])*left,
                        (1-skip[:, None])*(1-left),
                        np.broadcast_to(skip[:, None], left.shape)), axis=1)
        p = np.full(len(u), 1/len(u)) if prior is None else prior
        return cls(u, ell, p, k)

    def validate_history(self, history):
        if len(history) > 3:
            raise ValueError("at most three response slots")
        checked = []
        seen = set()
        for observation in history:
            if len(observation) != 2:
                raise ValueError("expected query and outcome")
            q, r = observation
            q = _integer(q, "query", 0, len(self.likelihood)-1)
            r = _integer(r, "outcome", 0, self.likelihood.shape[1]-1)
            if q in seen:
                raise ValueError("a question cannot be repeated")
            seen.add(q)
            checked.append((q, r))
        return tuple(checked)

    def mass(self, history=()):
        """Raw joint masses for algebraic checks; tiny products can underflow.

        Public decisions use posterior() and never interpret raw underflow as a
        logically impossible observation.
        """
        result = self.prior.copy()
        for q, r in sorted(self.validate_history(history)):
            result *= self.likelihood[q, r]
        return result

    def posterior(self, history=()):
        """Stable conditional distribution, ignoring constant evidence factors.

        Removing constant likelihoods before arithmetic gives exact skip
        invariance, including numerical ties. A zero constant is impossible;
        a very small positive constant is merely unlikely.
        """
        history = self.validate_history(history)
        factors = []
        for q, r in sorted(history):
            likelihood = self.likelihood[q, r]
            if np.all(likelihood == likelihood[0]):
                if likelihood[0] == 0:
                    raise ValueError("observed history is impossible under this model")
            else:
                factors.append(likelihood)
        if not factors:
            return self.prior.copy()
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.prior)
            for likelihood in factors:
                log_weights += np.log(likelihood)
        normalizer = logsumexp(log_weights)
        if not np.isfinite(normalizer):
            raise ValueError("observed history is impossible under this model")
        weights = np.exp(log_weights-normalizer)
        weights /= weights.sum()
        return weights

    def terminal(self, mass):
        return float(topk_sum(np.asarray(mass) @ self.utilities, self.k))

    def selection(self, history=()):
        scores = self.posterior(history) @ self.utilities
        return tuple(map(int, np.argsort(-scores, kind="stable")[:self.k]))

    def one_step_values(self, mass, available):
        ids = np.asarray(available, dtype=int)
        if len(ids) == 0:
            return np.empty(0)
        weighted = self.likelihood[ids] * np.asarray(mass)[None, None, :]
        return topk_sum(weighted @ self.utilities, self.k).sum(axis=1)

    def information_gain(self, mass, available):
        mass = np.asarray(mass)
        if mass.sum() <= 0:
            return np.zeros(len(available))
        p = mass/mass.sum()
        ell = self.likelihood[np.asarray(available, dtype=int)]
        predictive = ell @ p
        entropy = -xlogy(predictive, predictive).sum(axis=1)
        conditional = -xlogy(ell, ell).sum(axis=1) @ p
        return entropy-conditional


@dataclass(frozen=True)
class Plan:
    query: int | None
    selection: tuple[int, ...]
    expected_total: float
    current_total: float
    perfect_information_total: float
    nodes: int
    cache_hits: int
    exhaustive_over_available_queries: bool
    beam: int | None


def plan(model, history=(), *, budget=None, beam=None, max_nodes=1_000_000):
    """Floating-point Bellman enumeration, or an explicitly restricted search.

    With beam=None every remaining question is evaluated. This is still numeric
    floating-point arithmetic, not the exact rational oracle or Lean kernel.
    Otherwise only the
    beam best one-step questions at each internal state are expanded. The last
    level always evaluates all available questions. This is an approximation
    and never reported as a global optimum. An exceeded node limit raises;
    there is no hidden fallback or silent truncation.
    """
    history = model.validate_history(history)
    budget = (3-len(history) if budget is None
              else _integer(budget, "budget", 0, 3-len(history)))
    if beam is not None:
        beam = _integer(beam, "beam", 1, 10000)
    max_nodes = _integer(max_nodes, "max_nodes", 1, 100_000_000)
    initial_mass = model.posterior(history)
    all_queries = tuple(range(len(model.likelihood)))
    nodes = 0
    pruned = False

    @lru_cache(maxsize=None)
    def solve(observations, depth):
        nonlocal nodes, pruned
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError("explicit planning node limit exceeded")
        # Positive homogeneity permits rescaling at every node. Keep the
        # conditional distribution stable instead of multiplying tiny evidence
        # probabilities along an already observed path.
        mass = model.posterior(observations)
        current = model.terminal(mass)
        if depth == 0 or model.k == model.utilities.shape[1]:
            return current, None
        asked = {q for q, _ in observations}
        available = tuple(q for q in all_queries if q not in asked)
        if not available:
            return current, None
        one_step = model.one_step_values(mass, available)
        if depth == 1:
            best = int(np.argmax(one_step))
            if one_step[best] > current:
                return float(one_step[best]), available[best]
            return current, None
        if beam is not None and len(available) > beam:
            pruned = True
            order = np.argsort(-one_step, kind="stable")[:beam]
            available = tuple(available[int(i)] for i in order)
        best_value, best_query = current, None
        for q in available:
            value = 0.
            for r in range(model.likelihood.shape[1]):
                probability = float(model.likelihood[q, r] @ mass)
                if probability == 0:
                    continue
                extended = tuple(sorted(observations + ((q, r),)))
                value += probability*solve(extended, depth-1)[0]
            if value > best_value:
                best_value, best_query = value, q
        return best_value, best_query

    value, query = solve(tuple(sorted(history)), budget)
    upper = float(initial_mass @ topk_sum(model.utilities, model.k))
    return Plan(query, model.selection(history), value,
                model.terminal(initial_mass), upper,
                nodes, solve.cache_info().hits, not pruned, beam)
