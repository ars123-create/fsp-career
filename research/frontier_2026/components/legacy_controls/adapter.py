"""Frozen-core adapters, not new fitted methods or benchmark results.

AGPL-3.0 for this combined extension; see NOTICE.md and preserved PABBO license.
The original local cores and model weights remain unchanged. The frontier pool
uses semantic vector-pair order, which can change the tie break of a legacy
query enumeration. No method is given identity, hidden utility, or scenario.
"""
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from research.preference_topk.core import Belief, questions
from research.nonlinear_topk.core import Config, gaussian_queries, posterior, topk
from research.pabbo_topk.core import guarded_topk, load_model, recommend as neural_recommend
from ...finite import FiniteModel, tradeoff_pairs


ROOT = Path(__file__).resolve().parents[4]
FROZEN_GP = Config(residual_sd=.05, lengthscale=.25, balance_mean=.3, utility_scale=.05)
LINEAR_M = 9
LINEAR_SEED = 202610072
TEMPERATURE = .04
LAPSE = .02
METHODS = ("equal", "linear_evoi_3", "gp_topk", "gp_topk_guard",
           "pabbo_published", "pabbo_rl")
CHECKPOINTS = {
    "published": (
        "research/pabbo_topk/upstream/results/PABBO/PABBO_GP6D/ckpt.tar",
        "221f006026dfd6912da9b38c4b3e993e96a9fc8613a98356e1be57e497a7f75d"),
    "rl": ("research/pabbo_topk/checkpoints/rl-0600.pt",
           "f596a3e3d69212845f917ad62cfdcaec800d8a1cd679b37068137087070e9b56"),
}


@lru_cache(maxsize=2)
def _neural_model(variant):
    path, expected = CHECKPOINTS[variant]
    checkpoint = ROOT/path
    if sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise RuntimeError("frozen PABBO checkpoint has changed")
    # Both variants start from the published checkpoint in the legacy loader.
    published_path, published_sha = CHECKPOINTS["published"]
    if sha256((ROOT/published_path).read_bytes()).hexdigest() != published_sha:
        raise RuntimeError("published PABBO checkpoint has changed")
    with torch.random.fork_rng(devices=[]):
        model = load_model(None if variant == "published" else checkpoint).float()
    model.requires_grad_(False)
    return model


class _Session:
    def __init__(self, values):
        self.values = np.array(values, dtype=float, copy=True)
        self.pairs = tradeoff_pairs(self.values)
        self.values.setflags(write=False)
        # Reuse validation only: these constant utilities are not the prior.
        self.validator = FiniteModel.pairwise(
            np.ones((1, len(self.values))), self.pairs,
            k=min(5, len(self.values)), skip_probability=.15)
        self._query_ids = {
            (tuple(self.values[a]), tuple(self.values[b])): q
            for q, (a, b) in enumerate(self.pairs)
        }
        self.last_diagnostics = {}

    def _validate(self, history, k, acquisition, allowed):
        if not isinstance(history, (list, tuple)):
            raise ValueError("history must be a tuple/list of query-outcome pairs")
        history = self.validator.validate_history(history)
        if type(k) is not int or not 1 <= k <= 5:
            raise ValueError("k must be an integer in 1..5")
        if acquisition not in allowed:
            raise ValueError("unknown acquisition for this frozen control")
        k = min(k, len(self.values))
        asked = {q for q, _ in history}
        available = tuple(q for q in range(len(self.pairs)) if q not in asked)
        if len(history) == 3 or k == len(self.values) or acquisition == "none":
            available = ()
        return history, k, available

    def _index_history(self, history):
        return [(self.pairs[q], None if outcome == 2 else outcome)
                for q, outcome in history]

    def _vector_history(self, history):
        return [{"left": self.values[self.pairs[q][0]].tolist(),
                 "right": self.values[self.pairs[q][1]].tolist(),
                 "choice": None if outcome == 2 else outcome}
                for q, outcome in history]


class Equal(_Session):
    """The frozen equal-weights, zero-question baseline."""
    def recommend(self, history=(), *, acquisition="default", k=5):
        history, k, _ = self._validate(history, k, acquisition, {"default", "none"})
        self.last_diagnostics = {"slots": len(history), "questions": 0}
        return topk(self.values.mean(axis=1), k), None


class LinearEVOI(_Session):
    """Frozen simplex quadrature/response model with canonical available pairs."""
    def __init__(self, values):
        super().__init__(values)
        self.prior = Belief.linear(self.values, m=LINEAR_M, seed=LINEAR_SEED)

    def belief(self, history=()):
        history, _, _ = self._validate(history, 1, "default", {"default"})
        belief = self.prior
        for pair, outcome in self._index_history(history):
            belief = belief.observe(pair, outcome, temperature=TEMPERATURE, lapse=LAPSE)
        return belief

    def recommend(self, history=(), *, acquisition="default", k=5):
        history, k, available = self._validate(
            history, k, acquisition, {"default", "top5", "none"})
        belief = self.belief(history)
        selected = topk(belief.scores, k)
        self.last_diagnostics = {"slots": len(history), "hypotheses": len(belief.utilities),
                                 "model_gain": None}
        if not available:
            return selected, None
        allowed = {tuple(sorted(self.pairs[q])) for q in available}
        excluded = [p for p in combinations(range(len(self.values)), 2) if p not in allowed]
        options = questions(belief, k, asked=excluded,
                            temperature=TEMPERATURE, lapse=LAPSE)
        by_pair = {tuple(sorted(option.pair)): option for option in options}
        # Source formula enumerates index-oriented pairs. The scalar gain is
        # orientation-invariant; ties follow the frontier's global query IDs.
        chosen = max(available, key=lambda q: by_pair[tuple(sorted(self.pairs[q]))].expected_gain)
        self.last_diagnostics["model_gain"] = by_pair[tuple(sorted(self.pairs[chosen]))].expected_gain
        return selected, chosen


class GaussianTopK(_Session):
    """Frozen Laplace GP; optional NEW observed-feature dominance decision rule.

    guard=True keeps the original unconstrained Gaussian query acquisition.
    It is not a claim that this acquisition is exact for the guarded decision.
    """
    def __init__(self, values, *, guard=False):
        super().__init__(values)
        if type(guard) is not bool:
            raise ValueError("guard must be boolean")
        self.guard = guard
        self.config = FROZEN_GP

    def belief(self, history=()):
        history, _, _ = self._validate(history, 1, "default", {"default"})
        return posterior(self.values, self._index_history(history), self.config)

    def recommend(self, history=(), *, acquisition="default", k=5):
        history, k, available = self._validate(
            history, k, acquisition, {"default", "top5", "eubo", "none"})
        belief = self.belief(history)
        selected = (guarded_topk(self.values, belief.mean, k) if self.guard
                    else topk(belief.mean, k))
        self.last_diagnostics = {"slots": len(history), "guard": self.guard, "model_gain": None}
        if not available:
            return selected, None
        options = gaussian_queries(belief, k, [self.pairs[q] for q in available])
        attribute = "eubo" if acquisition == "eubo" else "gain"
        chosen = max(range(len(options)), key=lambda j: getattr(options[j], attribute))
        self.last_diagnostics["model_gain"] = options[chosen].gain*self.config.utility_scale
        return selected, available[chosen]


class PABBO(_Session):
    """Existing published-weight transfer or frozen RL adaptation, both top-5.

    'published' identifies the checkpoint, NOT an unchanged paper algorithm.
    Both modes use the existing empty-context/mean/dominance adapters.
    """
    def __init__(self, values, *, variant="published"):
        super().__init__(values)
        if variant not in CHECKPOINTS:
            raise ValueError("variant must be published or rl")
        self.variant = variant
        self._model = _neural_model(variant)

    def recommend(self, history=(), *, acquisition="default", k=5):
        history, k, available = self._validate(
            history, k, acquisition, {"default", "learned", "none"})
        result = neural_recommend(self._model, self.values, self._vector_history(history),
                                  k=k, strategy="learned" if available else "none")
        self.last_diagnostics = {"slots": result["slots_used"], "answers": result["answers"],
                                 "variant": self.variant,
                                 "checkpoint_sha256": CHECKPOINTS[self.variant][1]}
        question = result["question"]
        query = None
        if question is not None:
            key = (tuple(question["left"]), tuple(question["right"]))
            if key not in self._query_ids:
                raise RuntimeError("legacy PABBO returned a noncanonical vector pair")
            query = self._query_ids[key]
            if query not in available:
                raise RuntimeError("legacy PABBO returned a repeated/unavailable query")
        return tuple(result["selected"]), query


def make_session(method, values):
    """Only visible skill vectors enter the algorithm boundary."""
    if method == "equal":
        return Equal(values)
    if method == "linear_evoi_3":
        return LinearEVOI(values)
    if method in {"gp_topk", "gp_topk_guard"}:
        return GaussianTopK(values, guard=method == "gp_topk_guard")
    if method in {"pabbo_published", "pabbo_rl"}:
        return PABBO(values, variant="published" if method == "pabbo_published" else "rl")
    raise ValueError("unknown legacy control")
