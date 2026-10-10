"""Cold-start / fixed-anchor MoNNE adaptation, preserving authors' training.

Upstream copyright 2025 Hanyang Wang, MIT; original code and license in upstream/.
This module is a project adaptation, not reproduction of the full BOPE experiment.
Ensemble members are approximate utility hypotheses, not calibrated posterior draws.
"""
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
import io
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr
import torch

from ...finite import FiniteModel, plan, tradeoff_pairs

UPSTREAM = Path(__file__).resolve().parent/'upstream'
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
from bope.model import PairwiseRankingNet, utility_model_training_noiseless  # noqa:E402


@dataclass(frozen=True)
class Config:
    ensemble_size: int = 8
    seed: int = 2026100803
    temperature: float = .04
    lapse: float = .02
    skip_probability: float = .15


class MoNNE:
    """One synthetic session with a fixed pool and bounded question history."""

    def __init__(self, values, config=Config()):
        self.values = np.array(values, dtype=float, copy=True)
        self.pairs = tradeoff_pairs(self.values)
        self.config = config
        if (type(config.ensemble_size) is not int
                or not 1 <= config.ensemble_size <= 64):
            raise ValueError("ensemble_size must be 1..64")
        # Use the common validator without pretending this dummy utility is a prior.
        self.validator = FiniteModel.pairwise(np.ones((1, len(self.values))),
                                              self.pairs, k=min(5, len(self.values)),
                                              skip_probability=config.skip_probability)
        self.fit_logs = {}

    def hypotheses(self, history=()):
        history = self.validator.validate_history(history)
        # Canonical ordering makes fit independent of the order records are supplied.
        # Skip contributes no label and cannot refit or re-randomize the model.
        answered = tuple(sorted((q, r) for q, r in history if r != 2))
        return self._fit(answered)

    @lru_cache(maxsize=32)
    def _fit(self, answered):
        old_dtype = torch.get_default_dtype()
        log = io.StringIO()
        output = []
        with torch.random.fork_rng(devices=[]):
            try:
                torch.set_default_dtype(torch.double)
                values = torch.tensor(self.values, dtype=torch.double)
                anchors = torch.stack((torch.zeros(values.shape[1]),
                                       torch.ones(values.shape[1])))
                info = {"Y": [values[list(self.pairs[q])] for q, _ in answered],
                        "prefer": [1 if r == 0 else -1 for _, r in answered]}
                for member in range(self.config.ensemble_size):
                    seed = self.config.seed+member
                    torch.manual_seed(seed)
                    if answered:
                        with redirect_stdout(log):
                            model, _ = utility_model_training_noiseless(
                                values.shape[1], info, monotonic=True,
                                seed=seed, Bayesian=False)
                    else:
                        # The published training routine divides by zero at n=0.
                        # Its initialized architecture supplies an explicit prior.
                        model = PairwiseRankingNet(values.shape[1], True)
                    model.eval()
                    with torch.no_grad():
                        anchor_scores = model(anchors, anchors)[0].flatten()
                        raw = model(values, values)[0].flatten()
                        width = float(anchor_scores[1]-anchor_scores[0])
                        if not np.isfinite(width) or width <= 0:
                            raise FloatingPointError("MoNNE fixed anchors have no positive scale")
                        normalized = (raw-anchor_scores[0])/width
                        if (not torch.isfinite(normalized).all()
                                or float(normalized.min()) < -1e-8
                                or float(normalized.max()) > 1+1e-8):
                            raise FloatingPointError("MoNNE violates finite anchored utility bounds")
                        output.append(normalized.numpy())
            finally:
                torch.set_default_dtype(old_dtype)
        self.fit_logs[answered] = log.getvalue()
        result = np.array(output)
        result.setflags(write=False)
        return result

    def recommend(self, history=(), *, acquisition="top5", k=5):
        history = self.validator.validate_history(history)
        if type(k) is not int or not 1 <= k <= 5:
            raise ValueError("invalid k")
        k = min(k, len(self.values))
        if acquisition not in {"top5", "ieubo", "eubo", "information"}:
            raise ValueError("unknown acquisition")
        hypotheses = self.hypotheses(history)
        scores = hypotheses.mean(axis=0)
        from ...constraints import monotone_selection
        selected = monotone_selection(self.values, scores, k)
        asked = {q for q, _ in history}
        available = tuple(q for q in range(len(self.pairs)) if q not in asked)
        if len(history) == 3 or not available or k == len(self.values):
            return selected, None
        pairs = tuple(self.pairs[q] for q in available)
        if acquisition in {"top5", "information"}:
            # The ensemble already uses the observed labels. Do NOT condition
            # it on that history a second time. Reweight hypothetical next answers.
            model = FiniteModel.pairwise(
                hypotheses, pairs, k, temperature=self.config.temperature,
                lapse=self.config.lapse, skip_probability=self.config.skip_probability)
            if acquisition == "top5":
                q = plan(model, budget=1).query
                return selected, (None if q is None else available[q])
            gain = model.information_gain(model.prior, tuple(range(len(pairs))))
        else:
            indices = np.array(pairs)
            left, right = hypotheses[:, indices[:, 0]], hypotheses[:, indices[:, 1]]
            if acquisition == "eubo":
                gain = np.maximum(left, right).mean(axis=0)
            else:
                # Paper IEUBO replaces the joint predictive pair by independent
                # Gaussians with these marginal moments. This is an approximation.
                mu_a, mu_b = left.mean(axis=0), right.mean(axis=0)
                sd = np.sqrt(left.var(axis=0)+right.var(axis=0))
                safe = np.maximum(sd, 1e-15)
                z = (mu_a-mu_b)/safe
                gain = mu_a*ndtr(z)+mu_b*ndtr(-z)+safe*np.exp(-z*z/2)/np.sqrt(2*np.pi)
                gain = np.where(sd > 0, gain, np.maximum(mu_a, mu_b))
        return selected, available[int(np.argmax(gain))]
