"""Finite Bayesian model for synthetic skills and correlated item-family leakage.

Reuses scipy.optimize.linear_sum_assignment, inspected before implementation.
This is a research simulator; it is not a calibrated psychometric assessment.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.optimize import linear_sum_assignment


Query = tuple[int, int, int]  # candidate, skill, family


def entropy(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def assignment(weights: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Optional one-to-one invitations. Dummy columns let every employer abstain."""
    jobs, candidates = weights.shape
    augmented = np.concatenate([weights, np.zeros((jobs, jobs))], axis=1)
    rows, cols = linear_sum_assignment(augmented, maximize=True)
    pairs = [(int(j), int(i)) for j, i in zip(rows, cols)
             if i < candidates and weights[j, i] > 1e-12]
    return float(sum(weights[j, i] for j, i in pairs)), pairs


@dataclass
class Market:
    truth: np.ndarray
    requirements: np.ndarray
    compatibility: np.ndarray
    responses: np.ndarray
    assumed_leakage: float

    def weights(self, mastery: np.ndarray) -> np.ndarray:
        joint = np.prod(
            np.where(self.requirements[:, None, :], mastery[None, :, :], 1.0),
            axis=2,
        )
        return np.where(self.compatibility > 0,
                        self.compatibility * (2 * joint - 1), -1e6)

    def column(self, candidate: int, mastery: np.ndarray) -> np.ndarray:
        joint = np.prod(np.where(self.requirements, mastery, 1.0), axis=1)
        c = self.compatibility[:, candidate]
        return np.where(c > 0, c * (2 * joint - 1), -1e6)


def generate_market(seed: int, scenario: str, candidates=8, jobs=4,
                    skills=3, families=3, variants=3) -> Market:
    scenarios = {
        "clean": (0.0, 0.0),
        "leakage": (0.30, 0.30),
        "misspecified_leakage": (0.45, 0.20),
        "shifted_items": (0.30, 0.30),
        "correlated_skills": (0.30, 0.30),
    }
    actual_leakage, assumed = scenarios[scenario]
    # Same seed means same latent random numbers for every policy.
    rng = np.random.default_rng(seed)
    u = rng.random((candidates, skills))
    if scenario == "correlated_skills":
        ability = rng.choice([0.15, 0.85], size=(candidates, 1))
        truth = u < ability
    else:
        truth = u < 0.5
    requirements = np.zeros((jobs, skills), dtype=bool)
    for j in range(jobs):
        size = int(rng.choice([1, 2, 2]))
        requirements[j, rng.choice(skills, size=size, replace=False)] = True
    compatibility = rng.uniform(0.6, 1.0, (jobs, candidates))
    compatibility[rng.random((jobs, candidates)) < 0.15] = 0
    leaked = rng.random((candidates, skills, families)) < actual_leakage
    high = np.full((1, skills, families, 1), 0.90)
    low = np.full((1, skills, families, 1), 0.15)
    if scenario == "shifted_items":
        high = rng.uniform(0.70, 0.95, high.shape)
        low = rng.uniform(0.10, 0.30, low.shape)
    competent_or_leaked = truth[:, :, None, None] | leaked[:, :, :, None]
    likelihood = np.where(competent_or_leaked, high, low)
    responses = rng.random((candidates, skills, families, variants)) < likelihood
    return Market(truth, requirements, compatibility, responses, assumed)


class Belief:
    """Exact posterior over skill and per-family preknowledge, independently per skill.

    Marginalizing preknowledge afresh for every response would double-count a
    family. Keeping the joint latent state avoids that specific error.
    """

    def __init__(self, candidates=8, skills=3, families=3, leakage=0.0,
                 prior=0.5, high=0.90, low=0.15):
        self.states = np.array(list(product([0, 1], repeat=families + 1)))
        prior_weights = np.where(self.states[:, 0], prior, 1 - prior)
        prior_weights *= np.prod(
            np.where(self.states[:, 1:], leakage, 1 - leakage), axis=1)
        self.posterior = np.broadcast_to(prior_weights,
                                         (candidates, skills, len(prior_weights))).copy()
        self.response_probability = np.where(
            self.states[:, :1] | self.states[:, 1:], high, low).T
        self.counts = np.zeros((candidates, skills, families), dtype=int)
        self.additional = np.zeros(candidates, dtype=int)

    def mastery(self) -> np.ndarray:
        return self.posterior @ self.states[:, 0]

    def branches(self, batch: tuple[Query, ...]):
        """Exact joint predictive outcomes, including same-skill family dependence."""
        for outcomes in product([0, 1], repeat=len(batch)):
            probability = 1.0
            updates = {}
            for (i, k, f), y in zip(batch, outcomes):
                post = updates.get((i, k), self.posterior[i, k])
                likelihood = self.response_probability[f]
                if not y:
                    likelihood = 1 - likelihood
                unnormalized = post * likelihood
                p_y = float(unnormalized.sum())
                if p_y <= 0:
                    probability = 0
                    break
                probability *= p_y
                updates[(i, k)] = unnormalized / p_y
            if probability > 0:
                yield probability, updates

    def update(self, query: Query, correct: bool, additional=True):
        i, k, f = query
        likelihood = self.response_probability[f]
        if not correct:
            likelihood = 1 - likelihood
        posterior = self.posterior[i, k] * likelihood
        self.posterior[i, k] = posterior / posterior.sum()
        self.counts[i, k, f] += 1
        self.additional[i] += int(additional)

    def available(self, variants=3, candidate_limit=6) -> list[Query]:
        return [(int(i), int(k), int(f))
                for i, k, f in np.argwhere(self.counts < variants)
                if self.additional[i] < candidate_limit]

    def information_gain(self, query: Query, mastery: np.ndarray) -> float:
        i, k, _ = query
        future_entropy = sum(prob * entropy(post[(i, k)] @ self.states[:, 0])
                             for prob, post in self.branches((query,)))
        return float(entropy(mastery[i, k]) - future_entropy)


def value_of_information(batch: tuple[Query, ...], belief: Belief, market: Market,
                         mastery: np.ndarray, weights: np.ndarray,
                         current_value: float) -> float:
    expected_future = 0.0
    for probability, updates in belief.branches(batch):
        changed = {}
        for (i, k), posterior in updates.items():
            if i not in changed:
                changed[i] = mastery[i].copy()
            changed[i][k] = posterior @ belief.states[:, 0]
        future_weights = weights.copy()
        for i, candidate_mastery in changed.items():
            future_weights[:, i] = market.column(i, candidate_mastery)
        expected_future += probability * assignment(future_weights)[0]
    return float(expected_future - current_value)


POLICIES = ("no_extra", "random_naive", "information_family", "voi_naive",
            "voi_family", "pair_voi_family")


def choose_queries(policy: str, belief: Belief, market: Market,
                   remaining: int, rng) -> tuple[Query, ...]:
    available = belief.available(variants=market.responses.shape[3])
    if not available or remaining < 1:
        return ()
    if policy == "random_naive":
        return (available[int(rng.integers(len(available)))],)
    mastery = belief.mastery()
    information = {q: belief.information_gain(q, mastery) for q in available}
    if policy == "information_family":
        return (max(available, key=lambda q: (information[q], q)),)
    # Demand-related tests affect optional freshness evidence, never the public grade.
    relevant = [q for q in available if market.requirements[:, q[1]].any()]
    if not relevant:
        return ()
    weights = market.weights(mastery)
    current_value = assignment(weights)[0]
    values = {q: value_of_information((q,), belief, market, mastery, weights,
                                      current_value) for q in relevant}
    best_query = max(relevant, key=lambda q: (round(values[q], 12), information[q], q))
    best = (best_query,)
    rate = values[best_query]
    if policy != "pair_voi_family" or remaining < 2:
        return best
    for i in range(len(mastery)):
        if belief.additional[i] + 2 > 6:
            continue
        # A tractable candidate set: two most informative families per skill.
        shortlist = []
        for k in range(mastery.shape[1]):
            qs = [q for q in relevant if q[:2] == (i, k)]
            shortlist.extend(sorted(qs, key=lambda q: (information[q], q), reverse=True)[:2])
        for a, b in combinations(shortlist, 2):
            # Each tuple denotes the next unseen variant. The pair uses distinct families.
            if a[1:] == b[1:]:
                continue
            gain = value_of_information((a, b), belief, market, mastery, weights,
                                        current_value) / 2
            if gain > rate + 1e-12:
                best, rate = (a, b), gain
    return best


def run_policy(market: Market, policy: str, seed: int, budget=12) -> dict:
    n, k, families, _ = market.responses.shape
    leakage = market.assumed_leakage if "family" in policy else 0.0
    belief = Belief(n, k, families, leakage=leakage)
    for i in range(n):
        for skill in range(k):
            belief.update((i, skill, 0), bool(market.responses[i, skill, 0, 0]),
                          additional=False)
    rng = np.random.default_rng(seed + 991)
    trace = []
    if policy != "no_extra":
        while len(trace) < budget:
            batch = choose_queries(policy, belief, market, budget - len(trace), rng)
            if not batch:
                break
            for query in batch:
                i, skill, family = query
                variant = int(belief.counts[query])
                correct = bool(market.responses[i, skill, family, variant])
                belief.update(query, correct)
                trace.append([i, skill, family, variant, int(correct)])
    mastery = belief.mastery()
    _, pairs = assignment(market.weights(mastery))
    true_weights = market.weights(market.truth.astype(float))
    oracle, _ = assignment(true_weights)
    actual = float(sum(true_weights[j, i] for j, i in pairs))
    false_invites = sum(not market.truth[i, market.requirements[j]].all() for j, i in pairs)
    return {
        "regret": float(oracle - actual), "utility": actual,
        "oracle_utility": oracle,
        "brier": float(np.mean((mastery - market.truth) ** 2)),
        "false_invites": int(false_invites), "invitations": len(pairs),
        "answers": len(trace),
        "distinct_families": len({tuple(q[:3]) for q in trace}),
        "trace": trace,
    }
