"""P20: trained observable-state question shortlist plus bounded lookahead.

Known imitation/value-distillation pattern. No scientific novelty or global
optimality claim. Only the finite-model terminal reward ranks candidates.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import xlogy

from ...finite import FiniteModel, plan, tradeoff_pairs
from ...models import standard_prior


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = str(HERE/"artifact-v1/model.npz")
MODES = {"learned_lookahead", "random_lookahead", "teacher", "greedy",
         "student", "learned_kg", "random_kg", "guarded_kg"}
FAMILIES = ("linear", "geometric", "power_minus2", "power_plus2", "saturation", "bottleneck")
FEATURE_NAMES = (
    "remaining_slots", "used_slots", "answered_slots", "skip_slots", "k_fraction",
    "pool_size", "dimensions", "current_per_candidate", "posterior_entropy",
    "posterior_ess_fraction", "mean_posterior_sd", "topk_threshold",
) + tuple("family_"+f for f in FAMILIES) + (
    "a_skill0", "a_skill1", "a_skill2", "b_skill0", "b_skill1", "b_skill2",
    "abs_delta0", "abs_delta1", "abs_delta2", "distance",
    "a_mean", "b_mean", "a_sd", "b_sd", "covariance", "difference_mean", "difference_sd",
    "left_probability", "answer_entropy", "a_margin", "b_margin", "a_membership", "b_membership",
)


@dataclass(frozen=True)
class Config:
    mode: str = "learned_lookahead"
    m: int = 2
    prior_seed: int = 202610082099
    random_seed: int = 202610082052
    temperature: float = .04
    lapse: float = .02
    skip_probability: float = .15
    shortlist: int = 4
    teacher_pool: int = 8
    continuation_beam: int = 2
    checkpoint: str = DEFAULT_CHECKPOINT

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown P20 mode")
        for name, low, high in (("m", 1, 8), ("shortlist", 1, 100),
                                ("teacher_pool", 1, 100), ("continuation_beam", 1, 100),
                                ("prior_seed", 0, 2**63-1), ("random_seed", 0, 2**63-1)):
            if type(getattr(self, name)) is not int or not low <= getattr(self, name) <= high:
                raise ValueError(f"invalid {name}")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= self.lapse <= 1 or not 0 < self.skip_probability < 1:
            raise ValueError("invalid lapse/skip")


class Student:
    """Safe NumPy inference from numeric NPZ; no pickle or executable checkpoint."""
    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as saved:
            self.arrays = {key: saved[key].copy() for key in saved.files}
        required = {"w0", "b0", "w1", "b1", "w2", "b2", "mean", "scale", "target_scale", "format"}
        if set(self.arrays) != required or int(self.arrays["format"]) != 1:
            raise ValueError("incompatible P20 artifact format")
        shapes = {"w0": (32, len(FEATURE_NAMES)), "b0": (32,), "w1": (32, 32),
                  "b1": (32,), "w2": (1, 32), "b2": (1,),
                  "mean": (len(FEATURE_NAMES),), "scale": (len(FEATURE_NAMES),)}
        for name, shape in shapes.items():
            if self.arrays[name].shape != shape or not np.isfinite(self.arrays[name]).all():
                raise ValueError("invalid numeric weights/feature schema")
        if np.any(self.arrays["scale"] <= 0) or not float(self.arrays["target_scale"]) > 0:
            raise ValueError("invalid normalization")
        self.sha256 = sha256(self.path.read_bytes()).hexdigest()

    def predict(self, features):
        a = self.arrays
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("invalid observable feature matrix")
        x = (x-a["mean"])/a["scale"]
        for i in range(2):
            x = np.maximum(x @ a[f"w{i}"].T+a[f"b{i}"], 0)
        return (x @ a["w2"].T+a["b2"]).ravel()*float(a["target_scale"])


def action_values(model, history, roots, *, depth, beam):
    """Forced first query, followed by finite.py continuation (possibly pruned).

    Every last-step continuation checks all still available questions. The
    first roots are an explicit restriction; omitted roots have no certificate.
    """
    history = tuple(sorted(model.validate_history(history)))
    if not 1 <= depth <= 3-len(history):
        raise ValueError("invalid remaining lookahead depth")
    roots = tuple(roots)
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate roots")
    asked = {q for q, _ in history}
    for q in roots:
        model.validate_history(((q, 0),))
        if q in asked:
            raise ValueError("root already asked")
    weights = model.posterior(history)
    if depth == 1:
        return model.one_step_values(weights, roots), {"nodes": 0, "continuation_exhaustive": True}
    values, nodes, exhaustive = [], 0, True
    for q in roots:
        value = 0.
        for outcome in range(model.likelihood.shape[1]):
            probability = float(weights @ model.likelihood[q, outcome])
            if probability == 0:
                continue
            result = plan(model, history+((q, outcome),), budget=depth-1, beam=beam)
            value += probability*result.expected_total
            nodes += result.nodes
            exhaustive &= result.exhaustive_over_available_queries
        values.append(value)
    return np.array(values), {"nodes": nodes, "continuation_exhaustive": bool(exhaustive)}


class Session:
    def __init__(self, values, config=Config()):
        if not isinstance(config, Config):
            raise TypeError("config must be Config")
        self.values = np.array(values, dtype=float, copy=True)
        self.pairs = tradeoff_pairs(self.values)
        self.values.setflags(write=False)
        self.config = config
        self.prior = standard_prior(self.values, kind="mixed", m=config.m, seed=config.prior_seed)
        self.models, self.last_diagnostics = {}, {}
        self._student = None
        self._canonical = np.array(sorted(map(tuple, self.values)))

    def model(self, k=5):
        if type(k) is not int or not 1 <= k <= 100:
            raise ValueError("k must be an integer in 1..100")
        k = min(k, len(self.values))
        if k not in self.models:
            c = self.config
            self.models[k] = FiniteModel.pairwise(
                self.prior.utilities, self.pairs, k, prior=self.prior.probabilities,
                temperature=c.temperature, lapse=c.lapse, skip_probability=c.skip_probability)
        return self.models[k]

    def available(self, history, k=5):
        history = tuple(sorted(self.model(k).validate_history(history)))
        asked = {q for q, _ in history}
        return history, tuple(q for q in range(len(self.pairs)) if q not in asked)

    def _random(self, history):
        payload = self._canonical.tobytes()+json.dumps(sorted(history)).encode()
        seed = int.from_bytes(sha256(payload).digest()[:8], "little") ^ self.config.random_seed
        return np.random.default_rng(seed)

    def features(self, history=(), k=5):
        model = self.model(k)
        history, available = self.available(history, k)
        if not available:
            return available, np.empty((0, len(FEATURE_NAMES)))
        p, u = model.posterior(history), model.utilities
        mean = p @ u
        centered = u-mean
        variance = p @ (centered**2)
        sd = np.sqrt(np.maximum(variance, 0))
        cutoff = np.partition(mean, len(mean)-model.k)[len(mean)-model.k]
        particle_cutoff = np.partition(u, u.shape[1]-model.k, axis=1)[:, u.shape[1]-model.k]
        above = u > particle_cutoff[:, None]
        equal = u == particle_cutoff[:, None]
        # Share fractional membership among exact cutoff ties, including duplicates.
        membership = p @ (above+equal*((model.k-above.sum(axis=1))/equal.sum(axis=1))[:, None])
        tags = np.array(self.prior.families)
        context = np.array([
            (3-len(history))/3, len(history)/3,
            sum(r != 2 for _, r in history)/3, sum(r == 2 for _, r in history)/3,
            model.k/len(mean), len(mean)/100, self.values.shape[1]/3,
            model.terminal(p)/model.k, float(-xlogy(p, p).sum()/np.log(len(p))),
            float(1/np.sum(p*p)/len(p)), float(sd.mean()), cutoff,
        ]+[float(p[tags == f].sum()) for f in FAMILIES])
        indices = np.array([self.pairs[q] for q in available])
        a, b = indices[:, 0], indices[:, 1]
        xa, xb = np.zeros((len(a), 3)), np.zeros((len(a), 3))
        xa[:, :self.values.shape[1]], xb[:, :self.values.shape[1]] = self.values[a], self.values[b]
        cov = np.sum(p[:, None]*centered[:, a]*centered[:, b], axis=0)
        diff_sd = np.sqrt(np.maximum(variance[a]+variance[b]-2*cov, 0))
        left = model.likelihood[np.array(available), 0] @ p / (1-self.config.skip_probability)
        entropy = -xlogy(left, left)-xlogy(1-left, 1-left)
        result = np.column_stack((
            np.broadcast_to(context, (len(a), len(context))), xa, xb, np.abs(xa-xb),
            np.linalg.norm(xa-xb, axis=1), mean[a], mean[b], sd[a], sd[b], cov,
            mean[a]-mean[b], diff_sd, left, entropy, mean[a]-cutoff, mean[b]-cutoff,
            membership[a], membership[b],
        ))
        if result.shape[1] != len(FEATURE_NAMES) or not np.isfinite(result).all():
            raise FloatingPointError("invalid observable features")
        return available, result

    def predicted(self, history=(), k=5):
        available, features = self.features(history, k)
        if self._student is None:
            self._student = Student(self.config.checkpoint)
        return available, self._student.predict(features)

    def teacher_roots(self, history=(), k=5):
        history, available = self.available(history, k)
        count = min(len(available), self.config.teacher_pool)
        if count == len(available):
            return available
        model = self.model(k)
        kg = model.one_step_values(model.posterior(history), available)
        best = [available[int(i)] for i in np.argsort(-kg, kind="stable")[:min(4, count)]]
        remaining = [q for q in available if q not in best]
        random = self._random(history).choice(remaining, count-len(best), replace=False)
        return tuple(sorted(best+list(map(int, random))))

    def recommend(self, history=(), *, k=5):
        started = time.perf_counter()
        model = self.model(k)
        history, available = self.available(history, k)
        weights = model.posterior(history)
        scores = weights @ model.utilities
        chosen = tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], tuple(-self.values[i])))[:model.k])
        mode, depth = self.config.mode, 3-len(history)
        self.last_diagnostics = {
            "mode": mode, "slots": len(history), "remaining_depth": depth,
            "available_questions": len(available), "hypotheses": len(weights),
            "posterior_ess": float(1/np.sum(weights**2)),
            "teacher_global_optimality": False, "omitted_action_gap_certificate": None,
            "learned_utility_ranking": False,
        }
        if depth == 0 or not available or model.k == len(self.values):
            self.last_diagnostics["recommend_seconds"] = time.perf_counter()-started
            return chosen, None
        current = model.terminal(weights)
        # Main lookahead policies all use the same exhaustive final-slot control.
        if mode == "greedy" or (depth == 1 and mode in {"learned_lookahead", "random_lookahead", "teacher"}):
            roots = available
            values = model.one_step_values(weights, roots)
            checker = {"nodes": 0, "continuation_exhaustive": True}
        else:
            if mode == "teacher":
                roots = self.teacher_roots(history, k)
            elif mode.startswith("random"):
                roots = tuple(sorted(map(int, self._random(history).choice(
                    available, min(len(available), self.config.shortlist), replace=False))))
            else:
                ids, predicted = self.predicted(history, k)
                rank = np.argsort(-predicted, kind="stable")
                roots = tuple(sorted(ids[int(i)] for i in rank[:min(len(ids), self.config.shortlist)]))
                if mode == "student":
                    query = ids[int(rank[0])]
                    self.last_diagnostics.update({"shortlist": list(roots), "checked_questions": 0,
                                                  "pool_coverage": len(roots)/len(available),
                                                  "recommend_seconds": time.perf_counter()-started})
                    return chosen, query
                if mode == "guarded_kg":
                    kg = model.one_step_values(weights, available)
                    roots = tuple(sorted(set(roots) | {available[int(np.argmax(kg))]}))
            if mode in {"teacher", "learned_lookahead", "random_lookahead"}:
                values, checker = action_values(model, history, roots, depth=depth,
                                               beam=self.config.continuation_beam)
            else:
                values = model.one_step_values(weights, roots)
                checker = {"nodes": 0, "continuation_exhaustive": True}
        index = int(np.argmax(values))
        query = roots[index] if values[index] > current else None
        self.last_diagnostics.update({
            "shortlist": list(roots), "checked_questions": len(roots),
            "pool_coverage": len(roots)/len(available), "current_total": current,
            "chosen_checker_value": float(max(current, values[index])),
            "checker_depth": depth if mode in {"teacher", "learned_lookahead", "random_lookahead"} else 1,
            **checker, "recommend_seconds": time.perf_counter()-started,
        })
        return chosen, query
