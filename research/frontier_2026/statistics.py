"""Paired seed-cluster analysis; approximations and multiplicity are explicit.

SciPy supplies distributions and bootstrap resampling. Rows within a seed are
never treated as independent, and optimizer failures may not be discarded.
"""
from itertools import product
from math import ceil

import numpy as np
from scipy.stats import bootstrap, norm, t


def balanced_losses(rows, methods, families, seeds):
    """Return cluster × family × method, refusing missing/failed/duplicate data."""
    if any(len(set(items)) != len(items) for items in (methods, families, seeds)):
        raise ValueError('Unique labels required')
    lookup = {}
    wanted = set(product(seeds, families, methods))
    for row in rows:
        key = row['seed'], row['family'], row['method']
        if key not in wanted:
            continue
        if key in lookup:
            raise ValueError(f'Duplicate benchmark row: {key}')
        if row['status'] != 'PASS':
            raise ValueError(f'Failed method cannot be dropped from comparison: {key}')
        if row.get('constraint_status') != 'PASS':
            raise ValueError(f'Hard-constraint violation or unverified contract: {key}')
        value = row['regret']
        if not np.isfinite(value) or not 0 <= value <= 1+1e-12:
            raise ValueError(f'Invalid loss: {key}')
        lookup[key] = value
    missing = wanted-set(lookup)
    if missing:
        raise ValueError(f'Missing benchmark rows, first: {sorted(missing)[:3]}')
    return np.array([[[lookup[s, f, m] for m in methods] for f in families] for s in seeds])


def empirical_cvar(values, alpha=.95, *, axis=-1):
    """Mean upper (1-alpha) probability mass, with fractional boundary weight."""
    x = np.asarray(values, dtype=float)
    if not 0 <= alpha < 1 or x.shape[axis] < 1 or not np.isfinite(x).all():
        raise ValueError('Finite sample and alpha in [0,1) required')
    ordered = np.flip(np.sort(x, axis=axis), axis=axis)
    tail_mass = (1-alpha)*x.shape[axis]
    weights = np.clip(tail_mass-np.arange(x.shape[axis]), 0, 1)
    shape = [1]*x.ndim
    shape[axis] = len(weights)
    return np.sum(ordered*weights.reshape(shape), axis=axis)/tail_mass


def cluster_mean_interval(difference, *, alpha=.05, hypotheses=1):
    """Bonferroni t intervals on independent seed-cluster means.

Coverage is a large-sample/model approximation, not distribution-free proof.
The one-sided upper bound is used for preregistered superiority gates.
"""
    x = np.asarray(difference, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite independent cluster differences required')
    if not 0 < alpha < 1 or type(hypotheses) is not int or hypotheses < 1:
        raise ValueError('Invalid confidence family')
    n, mean = len(x), float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd/np.sqrt(n)
    one = float(t.ppf(1-alpha/hypotheses, n-1))*se
    two = float(t.ppf(1-alpha/(2*hypotheses), n-1))*se
    return {'mean': mean, 'cluster_sd': sd, 'standard_error': se, 'clusters': n,
            'upper_one_sided': mean+one, 'two_sided': [mean-two, mean+two],
            'alpha': alpha, 'multiplicity': hypotheses,
            'scope': 'Paired seed-cluster t approximation; no human-population coverage claim'}


def cvar_difference_interval(candidate, baseline, *, alpha=.95, confidence_alpha=.05,
                             hypotheses=1, resamples=9999, seed=202610084005):
    """Paired *cluster* bootstrap of tail loss under equal family weighting.

Resample seed indices, bringing every family for a seed together. The two
arrays must share shape C×F. A basic bootstrap upper interval is approximate.
"""
    a, b = np.asarray(candidate, float), np.asarray(baseline, float)
    if a.ndim != 2 or a.shape != b.shape or len(a) < 2 or not np.isfinite(a+b).all():
        raise ValueError('Matching finite cluster × family matrices required')
    if type(hypotheses) is not int or hypotheses < 1 or not 0 < confidence_alpha < 1:
        raise ValueError('Invalid confidence family')
    if type(resamples) is not int or resamples < 100:
        raise ValueError('At least 100 bootstrap resamples required')
    def statistic(indices):
        index = np.asarray(indices, dtype=int)
        return empirical_cvar(a[index].ravel(), alpha)-empirical_cvar(b[index].ravel(), alpha)
    measured = float(statistic(np.arange(len(a))))
    result = bootstrap((np.arange(len(a)),), statistic, vectorized=False,
                       n_resamples=resamples, method='basic',
                       confidence_level=1-confidence_alpha/hypotheses,
                       alternative='less', rng=np.random.default_rng(seed), batch=128)
    return {'difference': measured, 'upper_one_sided': float(result.confidence_interval.high),
            'bootstrap_standard_error': float(result.standard_error), 'clusters': len(a),
            'families': a.shape[1], 'cvar_alpha': alpha, 'resamples': resamples,
            'confidence_alpha': confidence_alpha, 'multiplicity': hypotheses, 'seed': seed,
            'scope': 'Basic paired seed-cluster bootstrap; approximate tail coverage'}


def prospective_clusters(cluster_sd, *, null_margin=-.001, alternative_mean=-.002, alpha=.05,
                         hypotheses=2, power=.9, minimum=256, maximum=4096):
    """Power to reject mean difference >= null_margin at alternative_mean.

    For gate upper < -.001, target alternative -.002 is .001 beyond the null.
    Power is NOT 90% if the true effect merely equals the practical null margin.
    """
    separation = null_margin-alternative_mean
    if (not np.isfinite(cluster_sd) or cluster_sd < 0 or not np.isfinite(separation)
            or separation <= 0):
        raise ValueError('Invalid variance/effect')
    if (not 0 < alpha < 1 or not .5 < power < 1 or type(hypotheses) is not int
            or hypotheses < 1):
        raise ValueError('Invalid confidence/power')
    if type(minimum) is not int or type(maximum) is not int or not 2 <= minimum <= maximum:
        raise ValueError('Cluster bounds must be integers with 2 <= minimum <= maximum')
    raw = ceil(((norm.ppf(1-alpha/hypotheses)+norm.ppf(power))*cluster_sd/separation)**2)
    chosen = min(max(raw, minimum), maximum)
    return {'normal_approximation_clusters': raw, 'chosen_clusters': chosen,
            'minimum': minimum, 'maximum': maximum, 'compute_cap_binding': raw > maximum,
            'target_power': power, 'null_margin': null_margin, 'alternative_mean': alternative_mean,
            'separation_beyond_null': separation,
            'cluster_sd': cluster_sd, 'alpha': alpha, 'hypotheses': hypotheses,
            'tail_metric_power': 'NOT_ESTABLISHED by this mean-loss calculation'}
