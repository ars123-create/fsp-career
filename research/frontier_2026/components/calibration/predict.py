"""Response-probability audit adapters, not fitted calibration or hiring scores.

Combined legacy-adapter integration is AGPL-3.0; see NOTICE.md. Underlying
finite/GP/ensemble components retain their authorship and original licenses.
"""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.special import xlogy

from ...finite import FiniteModel, tradeoff_pairs
from ...policies import ParticleSession
from ..constrained_gp.adapter import Session as ConstrainedGP
from ..distillation.adapter import Session as Distillation
from ..legacy_controls.adapter import Equal, GaussianTopK, LinearEVOI, PABBO, LAPSE, TEMPERATURE
from ..monne.adapter import MoNNE
from ..risk_decisions.adapter import Session as Risk
from research.nonlinear_topk.core import gaussian_queries


COMMON_QUERY_SEED = 202610084271
LEGACY_NEUTRAL_SKIP = .15


class PredictionUnavailable(ValueError):
    """The method has no declared response distribution, or mode is ambiguous."""
    status = "NOT_AVAILABLE"


@dataclass(frozen=True)
class PredictionReport:
    probabilities: np.ndarray | None
    status: str
    source: str
    reason: str | None = None
    asked_fraction: float | None = None
    scope: str = "Future response on fixed common pairs; may include already-asked pairs; no new answer collected"


def common_query_ids(values, *, max_queries=32, seed=COMMON_QUERY_SEED):
    """One fixed common canonical subset from X only, independent of any policy.

    It intentionally includes already-asked pairs. A diagnostic forecast means
    a further independent response to that pair, not permission to ask again.
    """
    pairs = tradeoff_pairs(values)
    if type(max_queries) is not int or not 1 <= max_queries <= 32:
        raise ValueError("max_queries must be 1..32")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("invalid fixed common-query seed")
    if len(pairs) <= max_queries:
        return tuple(range(len(pairs)))
    canonical = np.array(sorted(set(map(tuple, np.asarray(values, dtype=float)))), dtype="<f8")
    canonical[canonical == 0.] = 0.  # normalize signed zero in a semantic feature hash
    payload = np.asarray(canonical.shape, dtype="<i8").tobytes()+canonical.tobytes()
    nonce = int.from_bytes(sha256(payload).digest()[:8], "little") ^ seed
    rng = np.random.default_rng(nonce)
    return tuple(sorted(map(int, rng.choice(len(pairs), max_queries, replace=False))))


def _validated_queries(session, query_ids):
    if tuple(session.pairs) != tradeoff_pairs(session.values):
        raise ValueError("session pairs are not the common canonical observed pool")
    result = []
    for query in query_ids:
        if isinstance(query, (bool, np.bool_)) or not isinstance(query, (int, np.integer)) or not 0 <= query < len(session.pairs):
            raise ValueError("invalid common query id")
        result.append(int(query))
    if len(set(result)) != len(result):
        raise ValueError("duplicate diagnostic queries")
    return tuple(result)


def _probabilities(value, *, columns=3):
    p = np.array(value, dtype=float, copy=True)
    if (p.ndim != 2 or p.shape[1] != columns or not np.isfinite(p).all()
            or np.any(p < -1e-12) or np.any(p > 1+1e-12)
            or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError(f"expected Q by {columns} finite probability rows summing to one")
    # Only remove roundoff-sized edge/sum drift; this is not a calibration fit.
    np.clip(p, 0., 1., out=p)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _categorical(left, skip):
    left = np.asarray(left, dtype=float)
    skip = np.broadcast_to(np.asarray(skip, dtype=float), left.shape)
    return np.column_stack(((1-skip)*left, (1-skip)*(1-left), skip))


def _risk_method(session, explicit):
    method = explicit if explicit is not None else session.last_diagnostics.get("method")
    if method == "mmr":
        raise PredictionUnavailable("MMR defines a bounded-error version space, not a predictive probability")
    if method not in {"kl_dro", "cvar", "baseline_safe", "edgecut"}:
        raise PredictionUnavailable("Risk mode is unknown; supply risk_method explicitly before recommendation")
    return method


def prediction_report(session, history=(), query_ids=(), *, risk_method=None):
    """Return Q×3 forecasts with the model/approximation status made explicit.

    Errors in history, numerics or model fitting propagate. Only a genuinely
    unavailable predictive model is converted into NOT_AVAILABLE.
    """
    ids = _validated_queries(session, query_ids)
    history = tuple(history)
    asked = {observation[0] for observation in history if len(observation) == 2}
    asked_fraction = (sum(q in asked for q in ids)/len(ids)) if ids else None
    k = min(5, len(session.values))
    try:
        if isinstance(session, ParticleSession):
            model = session._model(k)
            history = model.validate_history(history)
            weights = session._weights(model, history)
            result = model.likelihood[list(ids)] @ weights
            fractional = session.config.policy == "fractional"
            status = "GENERALIZED_POSTERIOR" if fractional else "MODEL_PREDICTIVE"
            source = "actual categorical likelihood marginalized under " + ("power posterior" if fractional else "finite posterior")
        elif isinstance(session, ConstrainedGP):
            weights = session.posterior_weights(history)
            result = np.array([session.response_likelihood(q) @ weights for q in ids]).reshape((-1, 3))
            status, source = "APPROXIMATE_SNIS", "P06 fixed truncated-prior particles, actual probit channel and posterior weights"
        elif isinstance(session, Distillation):
            model = session.model(k)
            weights = model.posterior(history)
            result = model.likelihood[list(ids)] @ weights
            status, source = "MODEL_PREDICTIVE", "P20 finite posterior; student network does not forecast responses"
        elif isinstance(session, Risk):
            method = _risk_method(session, risk_method)
            model = session._model(k)
            weights = model.posterior(history)
            result = model.likelihood[list(ids)] @ weights
            status, source = "MODEL_PREDICTIVE", f"Risk {method}: nominal finite Bayes response model; not adversarial terminal weights"
        elif isinstance(session, LinearEVOI):
            belief = session.belief(history)
            left = [float(belief.likelihood(session.pairs[q], temperature=TEMPERATURE, lapse=LAPSE)
                          @ belief.probabilities) for q in ids]
            result = _categorical(left, LEGACY_NEUTRAL_SKIP)
            status, source = "MODEL_PREDICTIVE", "Frozen linear belief/logistic channel plus explicit neutral skip0.15 wrapper"
        elif isinstance(session, GaussianTopK):
            belief = session.belief(history)
            options = gaussian_queries(belief, k, [session.pairs[q] for q in ids], noise_variance=2.)
            result = _categorical([option.probability_left for option in options], LEGACY_NEUTRAL_SKIP)
            status, source = "APPROXIMATE_LAPLACE", "Frozen Gaussian belief; gaussian_queries probability_left/noise_variance2; neutral skip0.15 wrapper"
        elif isinstance(session, MoNNE):
            # The ensemble is already fitted on actual history: do not multiply
            # its historical likelihood a second time.
            utilities = session.hypotheses(history)
            c = session.config
            model = FiniteModel.pairwise(utilities, [session.pairs[q] for q in ids], k,
                                         temperature=c.temperature, lapse=c.lapse,
                                         skip_probability=c.skip_probability)
            result = model.likelihood @ model.prior
            status, source = "UNCALIBRATED_ENSEMBLE", "MoNNE fitted ensemble, equal member mixture through its declared logistic channel"
        elif isinstance(session, (Equal, PABBO)):
            session.validator.validate_history(history)
            raise PredictionUnavailable("Equal/PABBO policy scores do not define a response probability model")
        else:
            raise PredictionUnavailable("No reviewed predictive adapter for this session type")
    except PredictionUnavailable as exc:
        return PredictionReport(None, "NOT_AVAILABLE", type(session).__name__, str(exc), asked_fraction)
    result = _probabilities(result)
    result.setflags(write=False)
    return PredictionReport(result, status, source, asked_fraction=asked_fraction)


def predicted_probabilities(session, history, query_ids, *, risk_method=None):
    """Return Q×3 [left,right,skip], or raise PredictionUnavailable."""
    report = prediction_report(session, history, query_ids, risk_method=risk_method)
    if report.probabilities is None:
        raise PredictionUnavailable(report.reason)
    return report.probabilities


def expected_scores(predicted, true_probabilities):
    """Evaluator-only expected proper losses against supplied true Q×3 rows.

    Multiclass Brier is the unhalved [0,2] convention. Conditional binary Brier
    uses the conventional single positive-class [0,1] convention. No eps floor
    hides a zero-probability log-loss failure: positive truth against zero
    prediction gives +inf. Undefined conditional binary rows have a false mask.
    """
    p, t = _probabilities(predicted), _probabilities(true_probabilities)
    if p.shape != t.shape:
        raise ValueError("predicted and true probabilities must share shape")
    brier = 1+np.sum(p*p, axis=1)-2*np.sum(p*t, axis=1)
    logloss = -np.sum(xlogy(t, p), axis=1)
    entropy = -np.sum(xlogy(t, t), axis=1)
    answered_p, answered_t = p[:, :2].sum(axis=1), t[:, :2].sum(axis=1)
    valid = (answered_p > 0) & (answered_t > 0)
    pred_left = np.full(len(p), np.nan)
    true_left = np.full(len(p), np.nan)
    pred_left[valid] = p[valid, 0]/answered_p[valid]
    true_left[valid] = t[valid, 0]/answered_t[valid]
    a, b = pred_left, true_left
    binary_brier = b*(1-a)**2+(1-b)*a*a
    binary_logloss = -xlogy(b, a)-xlogy(1-b, 1-a)
    return {
        "categorical_expected_brier": brier,
        "categorical_expected_logloss": logloss,
        "categorical_probability_squared_error": np.sum((p-t)**2, axis=1),
        "categorical_irreducible_brier": 1-np.sum(t*t, axis=1),
        "categorical_truth_entropy": entropy,
        "categorical_excess_logloss_kl": logloss-entropy,
        "skip_expected_brier": t[:, 2]*(1-p[:, 2])**2+(1-t[:, 2])*p[:, 2]**2,
        "skip_expected_logloss": -xlogy(t[:, 2], p[:, 2])-xlogy(1-t[:, 2], 1-p[:, 2]),
        "skip_probability_squared_error": (p[:, 2]-t[:, 2])**2,
        "conditional_binary_valid": valid,
        "conditional_binary_predicted_left": pred_left,
        "conditional_binary_true_left": true_left,
        "conditional_binary_expected_brier": binary_brier,
        "conditional_binary_expected_logloss": binary_logloss,
        "conditional_binary_probability_squared_error": (a-b)**2,
        "true_answer_probability": answered_t,
    }
