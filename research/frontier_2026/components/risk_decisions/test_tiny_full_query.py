"""Tiny full risk questions: independent complete-slate and branch oracles."""
from itertools import combinations
import math

import numpy as np
import pytest

from research.frontier_2026.finite import FiniteModel, tradeoff_pairs
from .adapter import Config
from .decisions import baseline_violation_masses
from .tiny_full_query import Session, plan_one_step


METHODS = ("kl_dro", "cvar", "baseline_safe")


def fixture():
    u = np.array([[.9,.3,.8,.2,.6,.5], [.2,.9,.3,.8,.4,.5]])
    ell = np.array([[[.72,.09],[.18,.81],[.1,.1]],
                    [[.45,.27],[.45,.63],[.1,.1]],
                    [[.2,.6],[.6,.2],[.2,.2]]])
    return FiniteModel(u,ell,np.array([.4,.6]),5)


def binary_kl_risk(loss, weights, radius):
    """Independent primal probability-interval bisection; no entropic tilt."""
    if loss[0] == loss[1]:
        return float(loss[0])
    j = int(loss[1] > loss[0])
    a = float(weights[j])
    if a == 0.:
        return float(loss[1-j])
    if a == 1. or radius >= -math.log(a):
        return float(loss[j])
    left, right = a, 1.
    for _ in range(80):
        p = (left+right)/2
        divergence = p*math.log(p/a)+(1-p)*math.log((1-p)/(1-a))
        if divergence <= radius:
            left = p
        else:
            right = p
    p = (left+right)/2
    return float(p*loss[j]+(1-p)*loss[1-j])


def independent_terminal(u, weights, method, baseline):
    k = 5
    oracle = [sum(sorted(map(float,row),reverse=True)[:k]) for row in u]
    choices = []
    base = np.array([(o-sum(float(row[i]) for i in baseline))/k for o,row in zip(oracle,u)])
    for slate in combinations(range(u.shape[1]),k):
        loss = np.array([(o-sum(float(row[i]) for i in slate))/k for o,row in zip(oracle,u)])
        mean = float(loss @ weights)
        if method == "kl_dro":
            risk = binary_kl_risk(loss,weights,.1)
        elif method == "cvar":
            # Scalar Rockafellar-Uryasev minimization at all finite breakpoints.
            risk = min(float(t+sum(w*max(float(l)-t,0) for l,w in zip(loss,weights))/.35)
                       for t in loss)
        else:
            if sum(w for w,l,b in zip(weights,loss,base) if l > b+.03) > .3:
                continue
            risk = mean
        choices.append((risk,mean,slate))
    return min(choices)[0]


@pytest.mark.parametrize("method",METHODS)
def test_all_queries_and_branches_match_independent_terminal_oracle(method):
    model = fixture()
    baseline = (0,1,2,4,5)
    plan = plan_one_step(model,method=method,alpha=.65,baseline=baseline,
                         epsilon=.03,delta=.3,query_cost=.0001)
    current = independent_terminal(model.utilities,model.prior,method,baseline)
    expected = []
    for q in range(3):
        future = 0.
        for r in range(3):
            raw = model.prior*model.likelihood[q,r]
            probability = float(sum(raw))
            leaf = independent_terminal(model.utilities,raw/probability,method,baseline)
            future += probability*leaf
            record = plan.diagnostics["query_records"][q]["branches"][r]
            assert record["probability"] == pytest.approx(probability,abs=1e-14)
            assert record["risk"] == pytest.approx(leaf,abs=2e-12)
        expected.append(future+.0001)
    assert plan.current_risk == pytest.approx(current,abs=2e-12)
    assert plan.value_with_stop == pytest.approx(min(current,*expected),abs=2e-12)
    chosen = int(np.argmin(expected)) if min(expected) < current-1e-12 else None
    assert plan.query == chosen
    assert plan.diagnostics["full_risk_acquisition"] is True
    assert plan.diagnostics["global_three_step_optimal"] is False
    assert len(plan.diagnostics["query_records"]) == 3


@pytest.mark.parametrize("method",METHODS)
def test_cost_stop_neutral_skip_and_three_slots(method):
    model = fixture()
    kwargs = dict(method=method,baseline=(0,1,2,4,5))
    initial = plan_one_step(model,**kwargs)
    skipped = plan_one_step(model,((0,2),),**kwargs)
    assert skipped.selection == initial.selection
    assert skipped.current_risk == initial.current_risk
    assert skipped.query != 0
    assert len(skipped.diagnostics["query_records"]) == 2
    assert plan_one_step(model,query_cost=1.,**kwargs).query is None
    final = plan_one_step(model,((0,0),(1,1),(2,2)),**kwargs)
    assert final.query is None and final.diagnostics["query_records"] == []
    with pytest.raises(ValueError,match="repeated"):
        plan_one_step(model,((0,0),(0,2)),**kwargs)
    with pytest.raises(ValueError,match="three"):
        plan_one_step(model,((0,0),(1,1),(2,2),(3,0)),**kwargs)


def test_zero_likelihood_branches_do_not_create_a_posterior():
    old = fixture()
    ell = np.array([[[0.,0.],[.8,.8],[.2,.2]]])
    model = FiniteModel(old.utilities,ell,old.prior,5)
    result = plan_one_step(model,method="cvar")
    assert result.query is None
    assert result.diagnostics["query_records"][0]["branches"][0]["risk"] is None
    assert result.diagnostics["terminal_calls"] == 1
    with pytest.raises(ValueError,match="impossible"):
        plan_one_step(model,((0,0),),method="cvar")


@pytest.mark.parametrize("method",METHODS)
def test_feature_only_adapter_constraints_and_actual_full_query(method):
    x = np.random.default_rng(78113).uniform(0,1,(8,3))
    session = Session(x,Config(m=1,terminal_solver="enumerate"))
    selected,q = session.recommend(method=method)
    assert session.pairs == tradeoff_pairs(x)
    assert len(set(selected)) == 5
    assert session.last_diagnostics["terminal_search_complete"] is True
    assert session.last_diagnostics["query_search_complete"] is True
    assert len(session.last_diagnostics["query_records"]) == len(session.pairs)
    assert session.last_diagnostics["method_variant"].endswith("full_risk_query_tiny")
    excluded = set(range(8))-set(selected)
    assert not any(np.all(x[j]>=x[i]) and np.any(x[j]>x[i]) for i in selected for j in excluded)
    if method == "baseline_safe":
        assert session.last_diagnostics["returned_violation_mass"] <= session.config.safe_delta
    if q is not None:
        _, following = session.recommend(((q,2),),method=method)
        assert following != q


@pytest.mark.parametrize("method",METHODS)
def test_small_pool_cannot_ask_and_full_slates_remain_valid(method):
    x = np.random.default_rng(711).uniform(size=(4,2))
    selected,q = Session(x).recommend(method=method)
    assert set(selected)==set(range(4)) and q is None


def test_exact_zero_baseline_tolerance_and_explicit_caps():
    x = np.random.default_rng(7310).uniform(size=(8,3))
    session = Session(x,Config(m=1,terminal_solver="enumerate",safe_epsilon=0,safe_delta=0))
    selected,_ = session.recommend(method="baseline_safe")
    assert session.last_diagnostics["returned_violation_mass"] == 0.
    for record in session.last_diagnostics["query_records"]:
        for branch in record["branches"]:
            if branch["probability"]:
                w = session._model(5).posterior(((record["query"],branch["outcome"]),))
                assert baseline_violation_masses(session.prior.utilities,[branch["selection"]],5,
                                                  w,session.baseline(),0.)[0] == 0.
    with pytest.raises(ValueError,match="at most"):
        Session(np.zeros((9,2)))
    with pytest.raises(ValueError,match="tiny adapter requires"):
        Session(x,Config(m=3,terminal_solver="enumerate"))
    with pytest.raises(ValueError,match="tiny adapter requires"):
        Session(x,Config(m=1,terminal_solver="library"))
    with pytest.raises(ValueError,match="nonnegative"):
        Session(x,query_cost=-.1)
