"""Finite-pool monotone Gaussian prior with rejection/importance Monte Carlo.

Inspired by the linear-constraint GP representation of Benavoli & Azzimonti
(UAI 2024). This is a narrow project implementation, not the authors' SUN
posterior engine or their experiment reproduction. See REUSE_SOURCES.md.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr, xlogy

from ...finite import FiniteModel, tradeoff_pairs
from research.preference_topk.core import topk_sum


@dataclass(frozen=True)
class Config:
    particles: int = 2048
    max_proposals: int = 262144
    proposal_batch: int = 2048
    seed: int = 2026100841
    amplitude: float = .25
    lengthscale: float = .60
    covariance_jitter: float = 1e-10
    probit_noise: float = .10
    lapse: float = .02
    skip_probability: float = .15
    minimum_ess: float = 16.

    def __post_init__(self):
        for name, lower, upper in (("particles", 2, 65536),
                                   ("max_proposals", 2, 2097152),
                                   ("proposal_batch", 1, 8192),
                                   ("seed", 0, 2**63-1)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in {lower}..{upper}")
        if self.max_proposals < self.particles:
            raise ValueError("max_proposals must cover the requested particles")
        for name in ("amplitude", "lengthscale", "probit_noise"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("covariance_jitter", "lapse", "skip_probability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        if (isinstance(self.minimum_ess, bool) or not np.isfinite(self.minimum_ess)
                or not 1 <= self.minimum_ess <= self.particles):
            raise ValueError("minimum_ess must be in [1,particles]")


class SamplingError(RuntimeError):
    """Explicit bounded sampler failure; diagnostics survive failure."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class PosteriorDegeneracyError(RuntimeError):
    def __init__(self, effective_sample_size, minimum):
        super().__init__(f"importance ESS {effective_sample_size:.6g} < {minimum:g}")
        self.effective_sample_size = effective_sample_size


class Session:
    """Fixed observed skill pool, a fixed prior sample, and at most three answers.

    Outcomes are 0=left, 1=right, 2=skip. Every observed vector must contain
    1..3 skills in [0,1]. The latent utility itself is unbounded, not a hiring
    probability. Duplicate vectors share exactly the same latent utility.
    This constrains dominance only within this supplied pool.
    """

    def __init__(self, values, config=Config()):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config")
        self.config = config
        self.values = np.array(values, dtype=float, copy=True)
        self.pairs = tradeoff_pairs(self.values)
        self.values.setflags(write=False)
        unique, inverse, counts = np.unique(self.values, axis=0,
                                            return_inverse=True, return_counts=True)
        self._unique, self._inverse, self._counts = unique, inverse, counts
        self._mean = unique.mean(axis=1)
        squared_distance = ((unique[:, None, :] - unique[None, :, :])**2).sum(axis=2)
        covariance = config.amplitude**2 * np.exp(
            -squared_distance/(2*config.lengthscale**2))
        covariance += np.eye(len(unique))*config.covariance_jitter
        if not np.isfinite(covariance).all():
            raise ValueError("kernel parameters overflow the finite covariance")
        self._covariance = covariance
        dominates = (unique[:, None, :] >= unique[None, :, :]).all(axis=2)
        np.fill_diagonal(dominates, False)
        self._dominance = np.nonzero(dominates)
        # Reuse only the shared finite-history validator; this is not the GP prior.
        self._validator = FiniteModel.pairwise(
            np.ones((1, len(self.values))), self.pairs,
            k=min(5, len(self.values)), skip_probability=config.skip_probability)
        self._samples, self.sampling_diagnostics = self._draw_prior()
        self._samples.setflags(write=False)
        self.utilities = self._samples[:, inverse].copy()
        self.utilities.setflags(write=False)
        # Canonical expanded columns make MC reductions independent of input order,
        # including multiplicities of identical observed skill vectors.
        self._canonical_utilities = self._samples[:, np.repeat(np.arange(len(unique)), counts)]
        self._canonical_utilities.setflags(write=False)

    def _draw_prior(self):
        cfg = self.config
        try:
            chol = np.linalg.cholesky(self._covariance)
        except np.linalg.LinAlgError as exc:
            raise SamplingError("covariance is not numerically positive definite",
                                {"status": "FAIL", "proposed": 0}) from exc
        rng = np.random.default_rng(cfg.seed)
        retained, count, proposed, accepted = [], 0, 0, 0
        upper, lower = self._dominance
        while count < cfg.particles and proposed < cfg.max_proposals:
            size = min(cfg.proposal_batch, cfg.max_proposals-proposed)
            candidates = self._mean + rng.standard_normal((size, len(self._unique))) @ chol.T
            feasible = np.ones(size, dtype=bool)
            # Bound temporary allocations even for a dense 100-profile order.
            for start in range(0, len(upper), 128):
                hi, lo = upper[start:start+128], lower[start:start+128]
                feasible &= (candidates[:, hi] >= candidates[:, lo]).all(axis=1)
                if not feasible.any():
                    break
            valid = candidates[feasible]
            proposed += size
            accepted += len(valid)
            take = min(len(valid), cfg.particles-count)
            if take:
                retained.append(valid[:take])
                count += take
        acceptance = accepted/proposed
        diagnostics = {
            "status": "PASS" if count == cfg.particles else "FAIL",
            "requested_particles": cfg.particles, "retained_particles": count,
            "proposed": proposed, "accepted_in_all_proposals": accepted,
            "acceptance_rate": acceptance, "rejection_rate": 1-acceptance,
            "acceptance_mcse": float(np.sqrt(acceptance*(1-acceptance)/proposed)),
            "observed_constraints": len(upper), "unique_profiles": len(self._unique),
            "duplicate_profiles": len(self.values)-len(self._unique),
            "proposal_cap": cfg.max_proposals,
            "scope": "Finite observed pool only; IID rejection then SNIS approximation",
        }
        if count != cfg.particles:
            raise SamplingError("explicit rejection proposal cap reached", diagnostics)
        result = np.concatenate(retained)
        if not np.isfinite(result).all():
            raise SamplingError("nonfinite Gaussian sample", diagnostics)
        return result, diagnostics

    def _history(self, history):
        return tuple(sorted(self._validator.validate_history(history)))

    def _z(self, query):
        a, b = self.pairs[query]
        return (self.utilities[:, a]-self.utilities[:, b])/self.config.probit_noise

    def _log_answer(self, query, outcome):
        z = self._z(query)
        log_cdf = log_ndtr(z if outcome == 0 else -z)
        lapse = self.config.lapse
        if lapse == 0:
            return log_cdf
        if lapse == 1:
            return np.full(len(z), -np.log(2.))
        return np.logaddexp(np.log(lapse/2), np.log1p(-lapse)+log_cdf)

    @lru_cache(maxsize=64)
    def _posterior(self, canonical_history):
        cfg = self.config
        log_weight = np.zeros(cfg.particles)
        informative = False
        for query, outcome in canonical_history:
            if outcome == 2:
                if cfg.skip_probability == 0:
                    raise ValueError("observed skip is impossible under this model")
                continue
            if cfg.skip_probability == 1:
                raise ValueError("observed answer is impossible under this model")
            factor = self._log_answer(query, outcome)
            # Constant evidence, including skips, is omitted before arithmetic.
            if np.all(factor == factor[0]):
                if not np.isfinite(factor[0]):
                    raise ValueError("observed history has no finite particle likelihood")
            else:
                log_weight += factor
                informative = True
        if informative:
            normalizer = logsumexp(log_weight)
            if not np.isfinite(normalizer):
                raise ValueError("observed history has no finite particle likelihood")
            weights = np.exp(log_weight-normalizer)
            weights /= weights.sum()
        else:
            weights = np.full(cfg.particles, 1/cfg.particles)
        ess = float(1/np.sum(weights**2))
        if ess < cfg.minimum_ess:
            raise PosteriorDegeneracyError(ess, cfg.minimum_ess)
        weights.setflags(write=False)
        return weights

    def posterior_weights(self, history=()):
        """SNIS weights over the same fixed prior sample; never refit/resample."""
        return self._posterior(self._history(history))

    def hypotheses(self, history=()):
        """Return H×N prior utilities; use posterior_weights(history) with them.

        The history is validated and its ESS checked, but samples are unchanged.
        Equal-weight averaging of these samples is the prior, not the posterior.
        """
        self.posterior_weights(history)
        return self.utilities

    def response_likelihood(self, query):
        """Three categorical probabilities as a 3×H array, useful for auditing."""
        self._validator.validate_history(((query, 0),))
        cfg = self.config
        left = cfg.lapse/2+(1-cfg.lapse)*ndtr(self._z(query))
        return np.array(((1-cfg.skip_probability)*left,
                         (1-cfg.skip_probability)*(1-left),
                         np.full(cfg.particles, cfg.skip_probability)))

    @staticmethod
    def _k(k, size):
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= 100:
            raise ValueError("k must be an integer in 1..100")
        return min(int(k), size)

    def _selection(self, weights, k):
        mean = (weights @ self._samples)[self._inverse]
        # Higher skill-vector tie break respects dominance even on exact utility ties.
        order = sorted(range(len(mean)), key=lambda i: (-mean[i], tuple(-self.values[i])))
        return tuple(order[:k])

    def acquisition_values(self, history=(), *, acquisition="top5", k=5):
        """Canonical available query IDs and one-step empirical acquisition values.

        Values for top5 are total expected terminal SUM top-k, not gains.
        This evaluates all remaining pairs but is not full-horizon planning.
        """
        if acquisition not in {"top5", "information", "eubo"}:
            raise ValueError("unknown acquisition")
        history = self._history(history)
        k = self._k(k, len(self.values))
        weights = self.posterior_weights(history)
        asked = {q for q, _ in history}
        available = tuple(q for q in range(len(self.pairs)) if q not in asked)
        if len(history) == 3 or k == len(self.values):
            return (), np.empty(0)
        return available, self._acquisition(weights, available, acquisition, k)

    def _acquisition(self, weights, queries, acquisition, k, rows=None):
        cfg = self.config
        utilities = self._canonical_utilities if rows is None else self._canonical_utilities[rows]
        original = self.utilities if rows is None else self.utilities[rows]
        current = float(topk_sum(weights @ utilities, k))
        output = []
        for start in range(0, len(queries), 64):
            indices = np.array([self.pairs[q] for q in queries[start:start+64]])
            if not len(indices):
                continue
            left_u, right_u = original[:, indices[:, 0]].T, original[:, indices[:, 1]].T
            if acquisition == "eubo":
                output.extend(np.maximum(left_u, right_u) @ weights)
                continue
            left = cfg.lapse/2+(1-cfg.lapse)*ndtr((left_u-right_u)/cfg.probit_noise)
            if acquisition == "information":
                predict = left @ weights
                entropy = -xlogy(predict, predict)-xlogy(1-predict, 1-predict)
                conditional = (-xlogy(left, left)-xlogy(1-left, 1-left)) @ weights
                output.extend((1-cfg.skip_probability)*(entropy-conditional))
                continue
            # Unnormalized branches exploit positive homogeneity of SUM top-k.
            left_scores = (left*weights) @ utilities
            right_scores = ((1-left)*weights) @ utilities
            output.extend((1-cfg.skip_probability) * (
                topk_sum(left_scores, k)+topk_sum(right_scores, k)
            ) + cfg.skip_probability*current)
        return np.array(output, dtype=float)

    def recommend(self, history=(), *, acquisition="top5", k=5):
        k = self._k(k, len(self.values))
        weights = self.posterior_weights(history)
        selected = self._selection(weights, k)
        available, values = self.acquisition_values(history, acquisition=acquisition, k=k)
        if not available or self.config.skip_probability == 1:
            return selected, None
        best = int(np.argmax(values))
        if acquisition == "top5":
            current = float(topk_sum(weights @ self._canonical_utilities, k))
            if values[best] <= current:
                return selected, None
        elif acquisition == "information" and values[best] <= 0:
            return selected, None
        return selected, available[best]

    def diagnostics(self, history=(), *, query=None, k=5, batches=8):
        """Approximate MC error, not a quality or model-correctness certificate.

        Query batch error holds the supplied query fixed; it does not correct
        for choosing the largest estimated acquisition value among many queries.
        """
        weights = self.posterior_weights(history)
        mean = weights @ self._samples
        n = len(weights)
        variance = n/(n-1) * np.sum(weights[:, None]**2*(self._samples-mean)**2, axis=0)
        result = dict(self.sampling_diagnostics)
        result.update({
            "config": asdict(self.config), "history": list(self._history(history)),
            "posterior_ess": float(1/np.sum(weights**2)),
            "posterior_max_weight": float(weights.max()),
            "posterior_mean": mean[self._inverse].tolist(),
            "posterior_mean_mcse": np.sqrt(variance)[self._inverse].tolist(),
            "mcse_scope": "SNIS delta method; not calibration, ranking or hiring coverage",
        })
        if query is not None:
            self._validator.validate_history(((query, 0),))
            if type(batches) is not int or not 2 <= batches <= n//2:
                raise ValueError("batches must be 2..particles//2")
            k = self._k(k, len(self.values))
            estimates = []
            for rows in np.array_split(np.arange(n), batches):
                local = weights[rows]
                mass = float(local.sum())
                if mass == 0:
                    raise PosteriorDegeneracyError(0., 1.)
                estimates.append(float(self._acquisition(local/mass, (query,), "top5", k, rows)[0]))
            result.update({
                "fixed_query": int(query), "fixed_query_batch_values": estimates,
                "fixed_query_batch_mcse": float(np.std(estimates, ddof=1)/np.sqrt(batches)),
                "fixed_query_mcse_scope": "Fixed-query split-batch diagnostic; no selected-best-query coverage",
            })
        return result
