"""Absolute-vector history adapter for the pinned m6 finite model.

The utility families, QMC seed, response likelihood and one-step objective are
unchanged. Historical comparisons remain usable when the eligible pool changes.
Only the current pool is eligible for selection or a new comparison.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .constraints import monotone_selection
from .core import response_probability
from .finite import FiniteModel, plan, tradeoff_pairs
from .models import standard_prior


@dataclass(frozen=True)
class Result:
    selected: tuple[int, ...]
    baseline: tuple[int, ...]
    scores: np.ndarray
    question: tuple[int, int] | None
    gain: float
    family_mass: dict[str, float]


def pair_key(left, right):
    return tuple(sorted((tuple(left), tuple(right))))


@lru_cache(maxsize=2)
def _compile(rows, previous_pairs, k):
    values = np.asarray(rows, dtype=float)
    pairs = tradeoff_pairs(values)
    prior = standard_prior(values, m=6)
    keys = [pair_key(values[a], values[b]) for a, b in pairs]
    differences = [prior.utilities[:, a] - prior.utilities[:, b] for a, b in pairs]
    for key in previous_pairs:
        if key not in keys:
            past = standard_prior(np.asarray(key), m=6)
            differences.append(past.utilities[:, 0] - past.utilities[:, 1])
            keys.append(key)
    delta = np.asarray(differences) if differences else np.empty((0, len(prior.probabilities)))
    left = response_probability(delta, temperature=.04, lapse=.02)
    likelihood = np.stack((.85 * left, .85 * (1-left), np.full_like(left, .15)), axis=1)
    model = FiniteModel(prior.utilities, likelihood, prior.probabilities, k)
    return values, pairs, tuple(keys), prior.families, model


def recommend(values, answers=(), *, k=5):
    values = np.asarray(values, dtype=float)
    tradeoff_pairs(values)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if len(answers) > 3:
        raise ValueError("at most three comparison slots")
    checked, seen = [], set()
    for left, right, choice in answers:
        pair = np.asarray((left, right), dtype=float)
        if pair.shape != (2, values.shape[1]):
            raise ValueError("comparison skills do not match current pool")
        if not tradeoff_pairs(pair):
            raise ValueError("comparison must contain a skill trade-off")
        if choice not in ("left", "right", "skip"):
            raise ValueError("unknown answer")
        key = pair_key(left, right)
        if key in seen:
            raise ValueError("repeated comparison")
        seen.add(key)
        response = 2 if choice == "skip" else int(choice == "right")
        if tuple(left) != key[0] and response != 2:
            response = 1-response
        checked.append((key, response))
    k = min(k, len(values))
    values, pairs, keys, families, model = _compile(
        tuple(map(tuple, values)), tuple(sorted(seen)), k)
    history = tuple((keys.index(key), response) for key, response in checked)
    posterior = model.posterior(history)
    scores = posterior @ model.utilities
    selected = monotone_selection(values, scores, k)
    baseline = monotone_selection(values, model.prior @ model.utilities, k)
    question, gain = None, 0.
    if len(history) < 3 and k < len(values):
        decision = plan(model, history, budget=1, beam=4)
        if decision.query is not None:
            # Appended historical questions are already in history, never new.
            assert decision.query < len(pairs)
            question = pairs[decision.query]
            gain = max(0., decision.expected_total-decision.current_total)
    tags = np.asarray(families)
    mass = {name: float(posterior[tags == name].sum()) for name in sorted(set(families))}
    return Result(selected, baseline, scores, question, gain, mass)
