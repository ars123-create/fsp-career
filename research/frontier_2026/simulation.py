"""Synthetic respondent/evaluator, deliberately separate from algorithm APIs.

These utility families are stress models, not labels from recruiters. Utilities
use fixed all-zero/all-one anchors; no candidate-pool min/max rescaling.
"""
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
from scipy.special import expit, ndtr

FAMILIES = ("linear", "geometric", "power", "bottleneck", "saturation", "interaction",
            "choquet", "threshold", "min_linear", "max_linear", "smooth_neural",
            "answer_noise", "query_skips", "correlated_quantized")
VARIANTS = ('base', 'utility_square', 'utility_sqrt', 'probit_noise', 'heavy_noise',
            'many_skips', 'query_coupling', 'measurement_error', 'nonmonotone')


@dataclass(frozen=True)
class World:
    seed: int
    family: str
    values: np.ndarray
    truth: np.ndarray
    uniforms: np.ndarray
    skip_uniforms: np.ndarray
    temperature: float
    lapse: float
    description: dict


def make_world(seed, family, *, n=18, d=3, variant='base'):
    if family not in FAMILIES or not 1 <= d <= 3 or not 1 <= n <= 100:
        raise ValueError("unknown simulation setting")
    rng = np.random.default_rng(seed)
    x = rng.uniform(.05, 1, (n, d))
    if family == "correlated_quantized":
        common = rng.normal(size=(n, 1))
        x = np.clip(.65+.2*common+.16*rng.normal(size=(n, d)), .1, 1)
        x = np.round(10*x)/10
    w = rng.dirichlet(np.full(d, rng.choice([.4, 1., 3.])))
    pair = (0, 0) if d == 1 else tuple(rng.choice(d, 2, replace=False))
    mix = rng.uniform(.25, .75)
    power = rng.choice([-4., -1., .5, 3., 5.])
    rates = rng.uniform(.5, 8, d)
    knots = rng.uniform(.25, .8, d)
    sharpness = rng.uniform(8, 24, d)
    linear_weights = rng.dirichlet(np.ones(d), size=4)
    intercept = rng.uniform(0, .35, 4)
    choquet_weight = rng.dirichlet(np.ones(d+2))
    hidden_weight = rng.uniform(.2, 2, (d, 7))
    hidden_bias = rng.uniform(-2, .5, 7)
    readout = rng.dirichlet(np.ones(7))
    def utility(z):
        linear = z @ w
        safe = np.maximum(z, 1e-12)
        if family in {"linear", "answer_noise"}:
            return linear
        if family == "geometric":
            return np.exp(np.log(safe) @ w)
        if family == "power":
            return (safe**power @ w)**(1/power)
        if family in {"bottleneck", "query_skips"}:
            return mix*linear+(1-mix)*np.min(z, axis=1)
        if family == "saturation":
            return ((1-np.exp(-z*rates))/(1-np.exp(-rates))) @ w
        if family in {"interaction", "correlated_quantized"}:
            return mix*linear+(1-mix)*np.sqrt(z[:, pair[0]]*z[:, pair[1]])
        if family == "choquet":
            features = np.c_[z, np.min(z, axis=1), np.max(z, axis=1)]
            return features @ choquet_weight
        if family == "threshold":
            low, high = expit(-sharpness*knots), expit(sharpness*(1-knots))
            return ((expit(sharpness*(z-knots))-low)/(high-low)) @ w
        if family in {"min_linear", "max_linear"}:
            raw = z @ linear_weights.T+intercept
            if family == "min_linear":
                return np.min(raw, axis=1)-np.min(intercept)
            return np.max(raw, axis=1)-np.max(intercept)
        if family == "smooth_neural":
            low = expit(hidden_bias) @ readout
            high = expit(hidden_weight.sum(axis=0)+hidden_bias) @ readout
            return (expit(z @ hidden_weight+hidden_bias) @ readout-low)/(high-low)
        raise AssertionError("missing family")
    truth = utility(x)
    if not np.isfinite(truth).all() or np.any((truth < -1e-12) | (truth > 1+1e-12)):
        raise ArithmeticError("invalid utility generator")
    temperature, lapse = (.16, .15) if family == "answer_noise" else (.055, .03)
    world = World(seed, family, x, truth, rng.random(3), rng.random(3),
                 temperature, lapse,
                 {"n": n, "d": d, "weight_concentration_choices": [.4, 1., 3.],
                  "scale": "fixed zero/one feature anchors; not pool min-max",
                  "source": "synthetic; no independent human annotations"})
    if variant not in VARIANTS:
        raise ValueError('Unknown prespecified stress variant')
    detail = dict(world.description, variant=variant)
    if variant in {'utility_square', 'utility_sqrt'}:
        truth = truth**(2 if variant == 'utility_square' else .5)
        detail['scale_change'] = 'Monotone transform preserves oracle set but changes cardinal regret and response difficulty'
    elif variant == 'measurement_error':
        observed_rng = np.random.default_rng(seed ^ 202610087310)
        x = np.clip(x+observed_rng.normal(0, .06, x.shape), 0, 1)
        detail['measurement'] = 'Observed vectors perturbed by Gaussian sd .06; latent generator values unchanged'
    elif variant == 'nonmonotone':
        truth = 1-np.mean((x-.65)**2, axis=1)/(.65**2)
        detail['assumption_violation'] = 'Utility prefers a .65 target level; monotone model assumption intentionally false'
    elif variant == 'probit_noise':
        detail['response_link'] = 'probit_derivative_matched_at_zero'
    elif variant == 'heavy_noise':
        temperature, lapse = .2, .25
    elif variant == 'many_skips':
        detail['override_skip_probability'] = .6
    elif variant == 'query_coupling':
        detail['random_coupling'] = 'deterministic per unordered observed-vector pair and task seed'
    return replace(world, values=x, truth=truth, temperature=temperature, lapse=lapse, description=detail)


def response_probabilities(world, pair):
    """True categorical probabilities, only for respondent/evaluator use."""
    a, b = pair
    difference = world.truth[a]-world.truth[b]
    if world.description.get('response_link') == 'probit_derivative_matched_at_zero':
        base = ndtr(difference/(world.temperature*4/np.sqrt(2*np.pi)))
    else:
        base = expit(difference/world.temperature)
    probability = world.lapse/2+(1-world.lapse)*base
    if world.family == "query_skips":
        # Respondent difficulty uses latent utility differences; the policy only
        # sees skill vectors. It is NOT the P14 assumed feature-distance model.
        skip = .08+.55*np.exp(-8*abs(difference))
    else:
        skip = .08
    skip = world.description.get('override_skip_probability', skip)
    return np.array([(1-skip)*probability, (1-skip)*(1-probability), skip])


def response(world, pair, slot):
    if not 0 <= slot <= 2:
        raise ValueError("only three response slots")
    full = response_probabilities(world, pair)
    skip = full[2]
    probability = full[0]/(1-skip)
    if world.description.get('random_coupling'):
        observed = np.array(sorted((tuple(world.values[pair[0]]), tuple(world.values[pair[1]]))))
        digest = sha256(str(world.seed).encode()+observed.tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        choice_uniform, skip_uniform = rng.random(2)
        if tuple(world.values[pair[0]]) > tuple(world.values[pair[1]]):
            choice_uniform = 1-choice_uniform
    else:
        choice_uniform, skip_uniform = world.uniforms[slot], world.skip_uniforms[slot]
    choice = 2 if skip_uniform < skip else int(choice_uniform >= probability)
    return choice, float(probability), float(skip)


def evaluate(world, selected):
    selected = tuple(selected)
    k = min(5, len(world.values))
    if (len(selected) != k or len(set(selected)) != k
            or any(type(i) not in (int, np.int64, np.int32) or not 0 <= i < len(world.values)
                   for i in selected)):
        raise ValueError("invalid terminal selection")
    ideal = np.argsort(-world.truth, kind="stable")[:k]
    regret = (world.truth[ideal].sum()-world.truth[list(selected)].sum())/k
    threshold = world.truth[ideal[-1]]
    above = set(np.flatnonzero(world.truth > threshold))
    tied = set(np.flatnonzero(world.truth == threshold))
    # Maximal overlap with ANY exact-utility-optimal slate. Duplicate profiles
    # should not lose secondary overlap solely because their row index differs.
    overlap = (len(set(selected)&above)+min(k-len(above), len(set(selected)&tied)))/k
    excluded = set(range(len(world.values)))-set(selected)
    violations = sum(np.all(world.values[j] >= world.values[i])
                     and np.any(world.values[j] > world.values[i])
                     for i in selected for j in excluded)
    return {"regret": max(0., float(regret)),
            "oracle_overlap": len(set(ideal)&set(selected))/k,
            "oracle_overlap_index_tiebreak": len(set(ideal)&set(selected))/k,
            "oracle_overlap_tie_aware": overlap,
            "dominance_violations": int(violations)}
