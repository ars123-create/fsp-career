"""Independent closed forms and cluster-boundary counterexamples."""
import numpy as np
import pytest
from scipy.stats import t

from .statistics import (balanced_losses, cluster_mean_interval, cvar_difference_interval,
                         empirical_cvar, prospective_clusters)


def test_fractional_tail_mass():
    assert empirical_cvar([0, 1, 2, 3], .625) == pytest.approx((3+.5*2)/1.5)
    assert empirical_cvar([0, 1, 2, 3], 0) == 1.5
    assert empirical_cvar([0, 1, 2, 3], .95) == 3
    assert np.allclose(empirical_cvar([[0, 2], [2, 4]], .5, axis=0), [2, 4])


def test_mean_interval_closed_form_and_multiplicity():
    values = np.array([-2, -1, 0, 1, 2.])
    report = cluster_mean_interval(values)
    assert report['standard_error'] == pytest.approx(1/np.sqrt(2))
    assert report['upper_one_sided'] == pytest.approx(t.ppf(.95, 4)/np.sqrt(2))
    assert cluster_mean_interval(values, hypotheses=20)['upper_one_sided'] > report['upper_one_sided']


def test_pairing_and_seed_clusters_are_preserved_in_bootstrap():
    a = np.arange(24).reshape(6, 4)/30
    result = cvar_difference_interval(a+.01, a, resamples=199, seed=77)
    assert result['difference'] == pytest.approx(.01)
    assert result['upper_one_sided'] == pytest.approx(.01)
    assert result['bootstrap_standard_error'] < 1e-14


def test_zero_effect_intervals_and_power_cap():
    assert cluster_mean_interval(np.zeros(20))['two_sided'] == [0., 0.]
    assert prospective_clusters(.5, maximum=256)['compute_cap_binding'] is True
    assert prospective_clusters(0.)['chosen_clusters'] == 256


def fixture_rows():
    return [{'method': m, 'seed': s, 'family': f, 'status': 'PASS', 'constraint_status': 'PASS', 'regret': .1+s*.01}
            for m in ['a', 'b'] for s in [1, 2] for f in ['x', 'y']]


def test_balanced_cube_keeps_unit_of_independence():
    cube = balanced_losses(fixture_rows(), ['a', 'b'], ['x', 'y'], [1, 2])
    assert cube.shape == (2, 2, 2)
    assert np.allclose(cube[:, :, 0], cube[:, :, 1])


@pytest.mark.parametrize('defect', ['missing', 'failed', 'duplicate', 'nan', 'constraint'])
def test_invalid_rows_cannot_silently_disappear(defect):
    rows = fixture_rows()
    if defect == 'missing': rows.pop()
    elif defect == 'failed': rows[0]['status'] = 'FAIL'
    elif defect == 'duplicate': rows.append(rows[0])
    elif defect == 'nan': rows[0]['regret'] = float('nan')
    elif defect == 'constraint': rows[0]['constraint_status'] = 'FAIL'
    with pytest.raises(ValueError):
        balanced_losses(rows, ['a', 'b'], ['x', 'y'], [1, 2])


def test_power_requires_separation_beyond_practical_null_margin():
    with pytest.raises(ValueError):
        prospective_clusters(.01, null_margin=-.001, alternative_mean=-.001)
    result = prospective_clusters(.01, null_margin=-.001, alternative_mean=-.003)
    assert result['separation_beyond_null'] == .002


@pytest.mark.parametrize('bounds', [(-1,-1),(1,4096),(256,128),(2.5,10),(True,10)])
def test_invalid_power_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        prospective_clusters(.01, minimum=bounds[0], maximum=bounds[1])
