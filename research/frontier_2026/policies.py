"""Runnable project hypotheses over declared finite utility distributions.

These are known decision/information principles and explicit adaptations. They
are not 12 newly invented published algorithms. P IDs map to proposals/.
"""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.special import logsumexp, xlogy

from research.preference_topk.core import response_probability, topk_sum
from .finite import FiniteModel, plan, tradeoff_pairs
from .models import standard_prior, choquet_prior, lattice_prior
from .constraints import hypothesis_topk, monotone_selection


@dataclass(frozen=True)
class Config:
    prior: str = "mixed"
    policy: str = "kg"
    m: int = 4
    seed: int = 2026100802
    temperature: float = .04
    lapse: float = .02
    skip_probability: float = .15
    horizon: int = 1
    beam: int | None = 4
    eta: float = .5
    epsilon_per_candidate: float = .001


def _entropy_masses(weights):
    total = np.sum(weights, axis=-1)
    return -np.sum(xlogy(weights, weights), axis=-1)+xlogy(total, total)


class ParticleSession:
    """Algorithm sees features and answers only; simulator state stays outside."""
    def __init__(self, values, config=Config()):
        self.values = np.array(values, dtype=float, copy=True)
        self.pairs = tradeoff_pairs(self.values)
        self.config = config
        if config.policy not in {"kg", "none", "random", "entropy", "bald", "eubo",
                                 "membership", "set_entropy", "fractional",
                                 "easy", "answerability", "thompson", "class_robust",
                                 "latent_noise"}:
            raise ValueError("unknown policy")
        if not 0 < config.skip_probability < 1:
            raise ValueError("FSP adapters must admit a neutral skip outcome")
        if not 0 < config.eta <= 1 or not 0 <= config.epsilon_per_candidate <= 1:
            raise ValueError("invalid fractional/ease parameter")
        if config.prior in {"linear", "mixed"}:
            self.prior = standard_prior(self.values, kind=config.prior,
                                        m=config.m, seed=config.seed)
        elif config.prior == "choquet":
            self.prior = choquet_prior(self.values, m=config.m, seed=config.seed)
        elif config.prior == "lattice":
            self.prior = lattice_prior(self.values, count=2**config.m, seed=config.seed)
        else:
            raise ValueError("unknown prior")
        self._models = {}
        canonical = np.array(sorted(map(tuple, self.values)), dtype=float)
        self.random_seed = int.from_bytes(sha256(canonical.tobytes()).digest()[:8], "little") ^ config.seed
        self.last_diagnostics = {}

    def _model(self, k):
        if k in self._models:
            return self._models[k]
        c = self.config
        u, prior = self.prior.utilities, self.prior.probabilities
        if c.policy == "answerability":
            # Declared, unvalidated UX surrogate: many changing coordinates and
            # very small feature distances increase the chance of abstention.
            complexity = np.array([np.count_nonzero(self.values[a] != self.values[b])
                                   for a, b in self.pairs])
            separation = np.array([np.linalg.norm(self.values[a]-self.values[b])
                                   for a, b in self.pairs])
            skip = np.clip(c.skip_probability+.1*(complexity-1)+.15*np.exp(-5*separation), .02, .8)
        else:
            skip = c.skip_probability
        if c.policy == "latent_noise":
            # Four declared noise hypotheses, marginalized jointly with utility.
            # No simulator temperature or family is provided to this adapter.
            parameters = ((.025, .02), (.08, .02), (.025, .2), (.08, .2))
            models = [FiniteModel.pairwise(u, self.pairs, k, prior=prior,
                                          temperature=t, lapse=l, skip_probability=skip)
                      for t, l in parameters]
            model = FiniteModel(np.tile(u, (len(models), 1)),
                                np.concatenate([m.likelihood for m in models], axis=2),
                                np.tile(prior/len(models), len(models)), k)
        else:
            model = FiniteModel.pairwise(u, self.pairs, k, prior=prior,
                                        temperature=c.temperature, lapse=c.lapse,
                                        skip_probability=skip)
        self._models[k] = model
        return model

    def _weights(self, model, history):
        if self.config.policy != "fractional":
            return model.posterior(history)
        # P13 is a generalized/power posterior, explicitly not ordinary Bayes.
        with np.errstate(divide="ignore"):
            log_weights = np.log(model.prior)
            for q, r in sorted(history):
                likelihood = model.likelihood[q, r]
                if not np.any(likelihood > 0):
                    raise ValueError('history has zero probability under every utility hypothesis')
                if not np.all(likelihood == likelihood[0]):
                    log_weights += self.config.eta*np.log(likelihood)
        normalizer = logsumexp(log_weights)
        if not np.isfinite(normalizer):
            raise ValueError('fractional history has no finite posterior mass')
        weights = np.exp(log_weights-normalizer)
        return weights/weights.sum()

    def recommend(self, history=(), *, k=5):
        k = min(k, len(self.values))
        model = self._model(k)
        history = model.validate_history(history)
        weights = self._weights(model, history)
        u = model.utilities
        scores = weights @ u
        selected = monotone_selection(self.values, scores, k)
        c = self.config
        asked = {q for q, _ in history}
        available = tuple(q for q in range(len(self.pairs)) if q not in asked)
        self.last_diagnostics = {"hypotheses": len(weights),
                                 "ess": float(1/np.sum(weights**2)),
                                 "slots": len(history), "policy": c.policy}
        if c.policy == "membership":
            membership = np.zeros_like(u)
            order = hypothesis_topk(self.values, u, k)
            np.put_along_axis(membership, order, 1., axis=1)
            objective = FiniteModel(membership, model.likelihood, model.prior, k)
            selected = monotone_selection(self.values, weights @ membership, k)
        else:
            objective = model
        if len(history) == 3 or not available or k == len(self.values) or c.policy == "none":
            return selected, None
        if c.policy in {"kg", "latent_noise", "answerability", "membership"}:
            result = plan(objective, history, budget=min(c.horizon, 3-len(history)), beam=c.beam)
            self.last_diagnostics.update({"nodes": result.nodes,
                                          "exhaustive": result.exhaustive_over_available_queries})
            return selected, result.query
        if c.policy == "random":
            rng = np.random.default_rng(self.random_seed+len(history))
            return selected, int(rng.choice(available))
        ell = model.likelihood[np.array(available)]
        branches = ell*weights[None, None, :]
        probability = branches.sum(axis=2)
        if c.policy == "entropy":
            acquisition = _entropy_masses(probability)
        elif c.policy == "bald":
            acquisition = model.information_gain(weights, available)
        elif c.policy == "eubo":
            pairs = np.array([self.pairs[q] for q in available])
            acquisition = np.maximum(u[:, pairs[:, 0]], u[:, pairs[:, 1]]).T @ weights
        elif c.policy == "fractional":
            current = float(topk_sum(scores, k))
            next_weights = weights[None, None, :]*ell**c.eta
            normalizers = next_weights.sum(axis=2, keepdims=True)
            # Impossible branches contribute exactly zero. Do not evaluate
            # their nonexistent posterior via 0/0 and then multiply NaN by 0.
            np.divide(next_weights, normalizers, out=next_weights, where=normalizers > 0)
            acquisition = (probability*topk_sum(next_weights @ u, k)).sum(axis=1)-current
            # This expected gain can be negative: no Bayes martingale claim.
            if np.max(acquisition) <= 0:
                return selected, None
        elif c.policy == "easy":
            gains = model.one_step_values(weights, available)-model.terminal(weights)
            if np.max(gains) <= 0:
                return selected, None
            admissible = np.flatnonzero(gains >= np.max(gains)-k*c.epsilon_per_candidate)
            def burden(index):
                a, b = self.pairs[available[int(index)]]
                delta = np.abs(self.values[a]-self.values[b])
                return (np.count_nonzero(delta), -float(delta.sum()), int(index))
            return selected, available[int(min(admissible, key=burden))]
        elif c.policy == "set_entropy":
            membership = np.zeros_like(u, dtype=np.uint8)
            np.put_along_axis(membership, hypothesis_topk(self.values, u, k), 1, axis=1)
            _, labels = np.unique(membership, axis=0, return_inverse=True)
            categories = np.eye(int(labels.max())+1)[labels]
            initial = float(_entropy_masses(weights @ categories))
            acquisition = initial-_entropy_masses(branches @ categories).sum(axis=1)
        elif c.policy == "thompson":
            rng = np.random.default_rng(self.random_seed+len(history))
            h1 = int(rng.choice(len(weights), p=weights))
            first = set(hypothesis_topk(self.values, u[h1:h1+1], k)[0])
            # Finite capped draw; explicit KG control when draws do not disagree.
            differing = None
            for _ in range(32):
                h2 = int(rng.choice(len(weights), p=weights))
                second = set(hypothesis_topk(self.values, u[h2:h2+1], k)[0])
                differing = [q for q in available if
                             (self.pairs[q][0] in first-second and self.pairs[q][1] in second-first)
                             or (self.pairs[q][1] in first-second and self.pairs[q][0] in second-first)]
                if differing:
                    break
            pool = tuple(differing) if differing else available
            self.last_diagnostics["thompson_no_disagreement"] = not bool(differing)
            gain = model.one_step_values(weights, pool)
            return selected, pool[int(np.argmax(gain))]
        elif c.policy == "class_robust":
            families = np.array(self.prior.families)
            scores_by_family = []
            for family in sorted(set(families)):
                mass = weights*(families == family)
                if mass.sum() == 0:
                    continue
                mass /= mass.sum()
                scores_by_family.append(model.one_step_values(mass, available)-model.terminal(mass))
            acquisition = np.min(scores_by_family, axis=0)
        else:
            raise AssertionError("unhandled declared policy")
        if not np.isfinite(acquisition).all():
            raise FloatingPointError("non-finite query acquisition")
        return selected, available[int(np.argmax(acquisition))]
