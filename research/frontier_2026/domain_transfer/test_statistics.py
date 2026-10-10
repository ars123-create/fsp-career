"""Independent small scalar/Fraction oracles for the new stratified estimator."""
from fractions import Fraction
from itertools import product
from math import sqrt

import numpy as np
import pytest
from scipy.stats import t

from .statistics import (weighted_cvar,weighted_mean_interval,stratified_tail_bootstrap,
                         bootstrap_difference,_tails_for_counts)
from ..statistics import cluster_mean_interval


def exact_tail(values,masses,alpha=Fraction(19,20)):
    remaining=1-alpha;total=Fraction(0)
    for value,mass in sorted(zip(values,masses),reverse=True):
        take=min(remaining,mass);total+=take*value;remaining-=take
        if not remaining:break
    assert remaining==0
    return float(total/(1-alpha))


def test_weighted_fractional_boundary_and_permutation():
    x=[Fraction(1,10),Fraction(1,5),Fraction(2,5),Fraction(9,10)]
    p=[Fraction(3,5),Fraction(1,4),Fraction(3,25),Fraction(3,100)]
    assert exact_tail(x,p)==pytest.approx(.7)
    assert weighted_cvar(x,p)==pytest.approx(.7,abs=2e-14)
    assert weighted_cvar(x[::-1],p[::-1])==pytest.approx(.7,abs=2e-14)
    assert weighted_cvar(x,p,alpha=0)==pytest.approx(float(sum(a*b for a,b in zip(x,p))),abs=2e-14)


def test_welch_against_scalar_reference_and_single_stratum():
    xs=[[.1,.2,.4,.8],[-.1,.2,.3]];weights=[.3,.7]
    means=[sum(x)/len(x) for x in xs]
    variances=[sum((v-mu)**2 for v in x)/(len(x)-1) for x,mu in zip(xs,means)]
    parts=[w*w*v/len(x) for w,v,x in zip(weights,variances,xs)]
    se=sqrt(sum(parts));df=sum(parts)**2/sum(a*a/(len(x)-1) for a,x in zip(parts,xs))
    mean=sum(w*mu for w,mu in zip(weights,means))
    result=weighted_mean_interval(xs,weights)
    assert result['mean']==pytest.approx(mean,abs=2e-15)
    assert result['standard_error']==pytest.approx(se,abs=2e-15)
    assert result['welch_degrees_of_freedom']==pytest.approx(df,abs=2e-14)
    assert result['upper_one_sided']==pytest.approx(mean+t.ppf(1-.05/89,df)*se,abs=2e-14)
    one=weighted_mean_interval([xs[0]],[1],hypotheses=3)
    reference=cluster_mean_interval(xs[0],hypotheses=3)
    assert one['two_sided']==pytest.approx(reference['two_sided'],abs=2e-14)


def test_zero_variance_is_explicit_degenerate_interval():
    result=weighted_mean_interval([[.1]*3,[-.2]*3],[.25,.75])
    assert result['welch_degrees_of_freedom'] is None
    assert result['two_sided']==pytest.approx([-.125,-.125])


def test_all_100_two_stratum_count_patterns_against_fraction():
    # Two strata, three roots each, two correlated family outcomes per root.
    x=[Fraction(i,20) for i in [19,2,13,4,7,6,11,8,3,10,1,12]]
    p=[Fraction(3,10)/6]*6+[Fraction(7,10)/6]*6
    options=[v for v in product(range(4),repeat=3) if sum(v)==3]
    counts=np.array([a+b for a,b in product(options,options)])
    root_ids=np.repeat(np.arange(6),2)
    actual,fallback=_tails_for_counts(np.array(x,float),root_ids,np.array(p,float),counts,alpha=.95,prefix_fraction=.05)
    assert fallback>0 and len(actual)==100
    for result,count in zip(actual,counts):
        masses=[mass*int(count[root]) for mass,root in zip(p,root_ids)]
        assert result==pytest.approx(exact_tail(x,masses),abs=3e-14)


def test_bootstrap_pairs_methods_preserves_affine_tail_and_chunking():
    x=np.arange(2*5*3,dtype=float).reshape(2,5,3)/100
    losses=np.stack((x,2*x+.1,x),axis=-1)
    small=stratified_tail_bootstrap(losses,[.2,.8],resamples=127,seed=173,batch=1,prefix_fraction=.05)
    large=stratified_tail_bootstrap(losses,[.2,.8],resamples=127,seed=173,batch=31,prefix_fraction=1)
    assert small['distribution']==pytest.approx(large['distribution'],abs=3e-14)
    assert small['distribution'][:,1]==pytest.approx(2*small['distribution'][:,0]+.1,abs=5e-14)
    identical=bootstrap_difference(small,0,2)
    assert identical['upper_one_sided']==identical['difference']==identical['bootstrap_standard_error']==0
    diff=bootstrap_difference(small,0,1)
    sample=small['distribution'][:,0]-small['distribution'][:,1]
    point=float(small['point'][0]-small['point'][1])
    assert diff['upper_one_sided']==pytest.approx(2*point-np.quantile(sample,.05/89),abs=2e-15)


@pytest.mark.parametrize('weights',[[.2,.7],[-.1,1.1],[0,1],[np.nan,.5]])
def test_invalid_stratum_weights_fail(weights):
    with pytest.raises(ValueError):weighted_mean_interval([[1,2],[3,4]],weights)


@pytest.mark.parametrize('values,masses', [([.1,.2],[.1,.2]),([.1,np.nan],[.5,.5]),([.1,.2],[-.1,1.1]),([] ,[])])
def test_invalid_tail_input_fails(values,masses):
    with pytest.raises(ValueError):weighted_cvar(values,masses)


def test_missing_or_nonfinite_bootstrap_cases_fail():
    x=np.zeros((2,5,3,2));x[0,0,0,0]=np.nan
    with pytest.raises(ValueError):stratified_tail_bootstrap(x,[.2,.8],resamples=100)
