"""Alternative input controls over frozen application scores; synthetic only."""
from types import SimpleNamespace

import numpy as np

from app.preference_engine import response_probability
from app.preference_matching import posterior, prepare_preference, select_preference
from app.schemas import SearchInput

SKILLS = ("Python", "SQL", "API")
SIGMAS = (.05, .15, .30)
FIXED_PAIRS = ((0, 1), (0, 2), (1, 2))


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("three finite nonnegative weights required")
    if weights.sum() <= 0:
        raise ValueError("at least one positive weight required")
    return weights / weights.sum()


def weighted_topk(values, weights, k=5):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("finite candidate x three-skill matrix required")
    if np.any((values < 0) | (values > 1)) or not 1 <= k <= len(values):
        raise ValueError("normalized scores and valid k required")
    scores = np.average(values, axis=1, weights=normalize_weights(weights))
    return tuple(int(i) for i in np.argsort(-np.round(scores, 12), kind="stable")[:k])


def numeric_control(weights, *, step=None, sigma=0., z=None):
    """Noise settings are assumptions, never estimates of human precision."""
    result = normalize_weights(weights).copy()
    if step is not None:
        if not np.isfinite(step) or not 0 < step <= 1:
            raise ValueError("positive rounding step up to one required")
        result = np.floor(result / step + .5) * step
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("finite nonnegative noise scale required")
    if sigma:
        z = np.asarray(z, dtype=float)
        if z.shape != (3,) or not np.isfinite(z).all():
            raise ValueError("explicit three-coordinate noise required")
        result = np.maximum(result + sigma * z, 0.)
    return normalize_weights(result) if result.sum() > 0 else np.full(3, 1 / 3)


def utility(values, weights, *, geometric=False):
    values = np.asarray(values, dtype=float)
    if geometric:
        return np.exp(np.log(values) @ weights)
    return values @ weights


def simulate_answer(left, right, case, turn):
    latent = utility(np.array([left, right]), case["weights"],
                     geometric=case["scenario"] == "geometric")
    probability = float(response_probability(latent[0] - latent[1],
                                            case["temperature"], case["lapse"]))
    return ("left" if case["uniforms"][turn] < probability else "right"), probability


def app_control(case):
    """Call current adapter, including early stop and its allowed question pool."""
    candidates = [SimpleNamespace(
        user_id=f"p{i:03d}", achievements=[], grade="Middle",
        skills={skill: {"score": float(value * 100)}
                for skill, value in zip(SKILLS, row)},
    ) for i, row in enumerate(case["observed"])]
    ranked = [(candidate, (0., [], [])) for candidate in candidates]
    history, trace = [], []
    while True:
        query = SearchInput(specialization="backend", ranking={
            "mode": "preference", "comparisons": history,
        })
        prepare_preference(query)
        selected, report = select_preference(ranked, query)
        item = {"selected": [int(c.user_id[1:]) for c, _ in selected],
                "weights": [report.weights[s] for s in SKILLS],
                "answered": report.answered, "status": report.status}
        trace.append(item)
        if report.question is None:
            break
        if len(history) >= 3:
            raise AssertionError("adapter exceeded question budget")
        left = [report.question.left[s] / 100 for s in SKILLS]
        right = [report.question.right[s] / 100 for s in SKILLS]
        choice, probability = simulate_answer(left, right, case, len(history))
        item["question"] = {"left": left, "right": right, "choice": choice,
                            "probability_left": probability,
                            "expected_gain_points": report.question.expected_gain_points}
        history.append({"left": report.question.left, "right": report.question.right,
                        "choice": choice})
    return {"selected": trace[-1]["selected"], "weights": trace[-1]["weights"],
            "answers": len(history), "interaction": "adaptive_binary", "trace": trace}


def fixed_swing_control(case):
    """Compare three defined 50-to-100 score changes in a fixed order."""
    history, trace = [], []
    weights = np.full(3, 1 / 3)
    for turn, (a, b) in enumerate(FIXED_PAIRS):
        left, right = np.full(3, .5), np.full(3, .5)
        left[a], right[b] = 1., 1.
        choice, probability = simulate_answer(left, right, case, turn)
        history.append({"left": dict(zip(SKILLS, (left * 100).tolist())),
                        "right": dict(zip(SKILLS, (right * 100).tolist())), "choice": choice})
        query = SearchInput(specialization="backend", skills=list(SKILLS), ranking={
            "mode": "preference", "comparisons": history,
        })
        hypotheses, probabilities = posterior(query)
        weights = probabilities @ hypotheses
        trace.append({"left": left.tolist(), "right": right.tolist(), "choice": choice,
                      "probability_left": probability, "weights": weights.tolist(),
                      "selected": list(weighted_topk(case["observed"], weights))})
    return {"selected": trace[-1]["selected"], "weights": weights.tolist(), "answers": 3,
            "interaction": "fixed_binary", "trace": trace}


def metrics(selected, true_utility, k=5):
    truth = np.asarray(true_utility, dtype=float)
    selected = np.asarray(selected, dtype=int)
    if len(selected) != k or len(set(selected)) != k or np.any((selected < 0) | (selected >= len(truth))):
        raise ValueError("distinct in-range selection of size k required")
    ideal = np.argsort(-truth, kind="stable")[:k]
    discount = 1 / np.log2(np.arange(2, k + 2))
    dcg = float(((2 ** truth[selected] - 1) * discount).sum())
    idcg = float(((2 ** truth[ideal] - 1) * discount).sum())
    return {"regret_points": max(0., float(truth[ideal].sum() - truth[selected].sum())) * 100 / k,
            "ndcg_at_5": dcg / idcg if idcg else 1.,
            "oracle_overlap": len(set(selected) & set(ideal)) / k}


def run_controls(case):
    controls = {"equal": (np.full(3, 1 / 3), "none"),
                "direct_exact": (normalize_weights(case["weights"]), "oracle_parameters"),
                "direct_round_5pp": (numeric_control(case["weights"], step=.05), "three_numbers")}
    for sigma in SIGMAS:
        controls[f"direct_sigma_{int(sigma * 100):03d}"] = (
            numeric_control(case["weights"], sigma=sigma, z=case["noise_z"]), "three_numbers")
    results = {name: {"selected": list(weighted_topk(case["observed"], weights)),
                      "weights": weights.tolist(), "answers": 0 if name == "equal" else None,
                      "interaction": interaction}
               for name, (weights, interaction) in controls.items()}
    results["app_evoi"] = app_control(case)
    results["fixed_swing"] = fixed_swing_control(case)
    for result in results.values():
        result.update(metrics(result["selected"], case["true_utility"]))
        result["weight_rmse"] = float(np.sqrt(np.mean((np.array(result["weights"]) - case["weights"]) ** 2)))
    return results
