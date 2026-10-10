"""Constructed correctness probes; these are not development/heldout scores."""
from fractions import Fraction
from itertools import combinations, product
from math import comb

import numpy as np
import pytest
from scipy.optimize import linprog, minimize
from scipy.special import rel_entr

from research.frontier_2026.finite import tradeoff_pairs
from . import adapter
from .adapter import Config, Session, VersionSpaceEmpty
from .decisions import (baseline_violation_masses, cvar, edge_disagreement, edge_gains, kl_worst_case,
                        slate_losses, solve_milp, solve_slates, top_slate)


def primal_kl(loss, w, radius):
    """Independent primal probability optimizer, not exponential-tilt code."""
    candidates = []
    for start in (w.copy(), np.full(len(w), 1/len(w))):
        result = minimize(lambda p: -p @ loss, start, jac=lambda p: -loss,
                          bounds=[(1e-14, 1.)]*len(w), method="SLSQP",
                          constraints=[{"type": "eq", "fun": lambda p: p.sum()-1,
                                        "jac": lambda p: np.ones(len(p))},
                                       {"type": "ineq", "fun": lambda p: radius-rel_entr(p, w).sum(),
                                        "jac": lambda p: -np.log(p/w)-1}],
                          options={"ftol": 1e-12, "maxiter": 2000})
        if result.success and abs(result.x.sum()-1) < 1e-9 and rel_entr(result.x,w).sum() <= radius+1e-8:
            # An optimizer success flag alone was empirically insufficient for
            # one tied-loss example; require independent KKT stationarity too.
            matrix = np.column_stack((np.log(result.x/w)+1, np.ones(len(w))))
            multiplier = np.linalg.lstsq(matrix,loss,rcond=None)[0]
            residual = np.max(np.abs(matrix @ multiplier-loss))
            if residual < 1e-5 and multiplier[0] >= -1e-8:
                candidates.append(result)
    assert candidates, "independent primal solver did not pass feasibility and KKT checks"
    result = min(candidates,key=lambda result: result.fun)
    return -float(result.fun)


@pytest.mark.parametrize("radius", [.005, .05, .2])
@pytest.mark.parametrize("loss", [np.array([.1,.4,.9]), np.array([-.5,.2,.2])])
def test_kl_against_independent_primal(radius, loss):
    w = np.array([.2,.3,.5])
    assert kl_worst_case(loss, w, radius) == pytest.approx(primal_kl(loss, w, radius), abs=1e-8)


def test_kl_support_limits_and_affine_units():
    w = np.array([.2,.8,0.])
    loss = np.array([.1,.8,100.])
    assert kl_worst_case(loss, w, 0) == pytest.approx(.66)
    assert kl_worst_case(loss, w, 10) == .8
    assert kl_worst_case(np.ones(3), w, .1) == 1
    small = kl_worst_case(loss, w, .04)
    assert kl_worst_case(5*loss-2, w, .04) == pytest.approx(5*small-2, abs=1e-10)


def independent_cvar(loss, w, alpha):
    # A finite piecewise-linear threshold optimum occurs at an observed loss.
    return min(float(t+sum(p*max(float(x)-t, 0) for p,x in zip(w,loss))/(1-alpha))
               for t in set(map(float, loss)))


@pytest.mark.parametrize("alpha", [0., .3, .7, .95])
def test_cvar_discrete_atoms_independent_threshold_and_lp(alpha):
    loss = np.array([0.,1.,2.])
    w = np.array([.2,.6,.2])
    # LP has a free threshold and three positive-part variables.
    a = np.column_stack((-np.ones(3), -np.eye(3)))
    result = linprog(np.r_[1.,w/(1-alpha)], A_ub=a, b_ub=-loss,
                     bounds=[(None,None)]+[(0,None)]*3, method="highs")
    assert result.success
    assert cvar(loss,w,alpha) == pytest.approx(result.fun, abs=1e-12)
    assert cvar(loss,w,alpha) == pytest.approx(independent_cvar(loss,w,alpha), abs=1e-12)
    if alpha == .7:
        assert cvar(loss,w,alpha) == pytest.approx(float(Fraction(5,3)), abs=1e-12)


def brute_loss(u, slate, k):
    # Deliberately scalar, independent of production incidence/matrix code.
    return np.array([(sum(sorted(row)[-k:])-sum(row[i] for i in slate))/k for row in u])


def brute_terminal(u,w,k,method,alpha,active,baseline,epsilon,delta):
    best = np.inf
    baseline_loss = brute_loss(u,baseline,k)
    for slate in combinations(range(u.shape[1]),k):
        loss = brute_loss(u,slate,k)
        if method == "cvar":
            value = independent_cvar(loss,w,alpha)
        elif method == "mmr":
            value = max(loss[h] for h in range(len(u)) if active[h])
        else:
            violation = sum(w[h] for h in range(len(u)) if loss[h] > baseline_loss[h]+epsilon)
            if violation > delta:
                continue
            value = sum(w[h]*loss[h] for h in range(len(u)))
        best = min(best,value)
    return float(best)


@pytest.mark.parametrize("method", ["cvar","mmr","baseline_safe"])
@pytest.mark.parametrize("n,k", [(6,2),(7,5),(5,5),(8,3)])
def test_milp_and_enumeration_against_independent_complete_slates(method,n,k):
    rng = np.random.default_rng(481+n+k)
    u = rng.uniform(-.5,1.,(4,n))
    w = np.array([.1,.2,.3,.4])
    active = np.array([True,False,True,True])
    baseline = tuple(range(k))
    options = dict(alpha=.7, active=active, baseline=baseline, epsilon=.03, delta=.25)
    expected = brute_terminal(u,w,k,method,**options)
    result,status = solve_milp(u,w,k,method,**options)
    assert result is not None, status
    assert status['terminal_search_complete'] and status['solver_status'] == 0
    assert result.risk == pytest.approx(expected,abs=2e-7)
    enumerated = solve_slates(u,w,k,method,combinations(range(n),k),complete=True,**options)
    assert enumerated.risk == pytest.approx(expected,abs=1e-12)
    assert len(result.selection) == len(set(result.selection)) == k


def test_kl_complete_slate_optimizer_against_independent_primal():
    u = np.array([[.1,.8,.7,.2],[.9,.3,.2,.6],[.4,.4,.7,.5]])
    w = np.array([.2,.3,.5])
    choices = list(combinations(range(4),2))
    independent = [primal_kl(brute_loss(u,s,2),w,.1) for s in choices]
    answer = solve_slates(u,w,2,'kl_dro',choices,radius=.1,complete=True)
    assert answer.risk == pytest.approx(min(independent),abs=1e-8)


def direct_edges(u,w,slates,k):
    total = 0.
    for h,j in combinations(range(len(u)),2):
        loss_h = (sum(sorted(u[h])[-k:])-sum(u[h,i] for i in slates[j]))/k
        loss_j = (sum(sorted(u[j])[-k:])-sum(u[j,i] for i in slates[h]))/k
        total += w[h]*w[j]*(loss_h+loss_j)/2
    return total


def test_edge_algebra_against_independent_quadratic_oracle():
    rng = np.random.default_rng(618)
    u = rng.uniform(0,1,(7,8))
    w = rng.dirichlet(np.ones(7))
    k = 5
    slates = [tuple(np.argsort(-row)[:k]) for row in u]
    z = np.zeros_like(u)
    for h,s in enumerate(slates):
        z[h,list(s)] = 1
    likelihood = rng.dirichlet(np.ones(3),size=(5,7)).transpose(0,2,1)
    current = direct_edges(u,w,slates,k)
    assert edge_disagreement(u,w,z,k) == pytest.approx(current,abs=1e-12)
    expected = []
    for q in range(5):
        value = current
        for r in range(3):
            mass = w*likelihood[q,r]
            value -= mass.sum()*direct_edges(u,mass/mass.sum(),slates,k)
        expected.append(value)
    assert edge_gains(u,w,likelihood,z,k) == pytest.approx(expected,abs=1e-12)


X = np.array([[.9,.1],[.8,.2],[.7,.3],[.6,.4],[.4,.6],[.3,.7],[.2,.8],[.1,.9]])
METHODS = ['kl_dro','cvar','mmr','baseline_safe','edgecut']


@pytest.mark.parametrize('method',METHODS)
def test_adapter_three_slots_skip_and_global_indices(method):
    session = Session(X,Config(m=1))
    assert session.pairs == tradeoff_pairs(X)
    before,_ = session.recommend(method=method)
    after,next_q = session.recommend(((0,2),),method=method)
    assert after == before
    assert next_q is None or next_q != 0
    final,q = session.recommend(((0,2),(1,2),(2,2)),method=method)
    assert final == before and q is None
    assert len(final) == len(set(final)) == 5
    with pytest.raises(ValueError):
        session.recommend(((0,2),(0,1)),method=method)
    with pytest.raises(ValueError):
        session.recommend(((0,2),(1,2),(2,2),(3,2)),method=method)
    with pytest.raises(ValueError):
        session.recommend(((0,3),),method=method)


@pytest.mark.parametrize('n',[1,3,5])
@pytest.mark.parametrize('method',METHODS)
def test_no_question_when_only_one_terminal_slate(n,method):
    session = Session(X[:n],Config(m=1))
    selected,q = session.recommend(method=method)
    assert selected == tuple(range(n)) and q is None


def test_version_error_budget_is_counted_not_bayesian():
    session = Session(X,Config(m=1,version_error_budget=1))
    _,q = session.recommend(method='mmr')
    assert q is not None, 'a first answer uses residual error budgets even without immediate elimination'
    history = ((q,0),((q+1)%len(session.pairs),1))
    u = session.prior.utilities
    expected = np.ones(len(u),dtype=bool)
    for h,row in enumerate(u):
        violations = 0
        for query,r in history:
            a,b = session.pairs[query]
            violations += int(row[a] < row[b]) if r == 0 else int(row[b] < row[a])
        expected[h] = violations <= 1
    assert np.array_equal(session.version_space(history),expected)
    assert np.array_equal(session.version_space(((q,2),)),np.ones(len(u),dtype=bool))


def test_empty_version_space_is_reported_without_prior_reset():
    session = Session(np.array([[.9,.1],[.6,.4],[.1,.9]]),Config(prior='linear',m=1,version_error_budget=0))
    assert len(session.pairs) == 3
    empty = []
    for outcome in product(range(2),repeat=3):
        history = tuple(enumerate(outcome))
        if np.all(session._version_counts(history,3)>0):
            empty.append(history)
    assert empty
    with pytest.raises(VersionSpaceEmpty):
        session.recommend(empty[0],method='mmr')


@pytest.mark.parametrize('method',METHODS)
def test_no_excluded_strict_skill_dominator(method):
    values = np.array([[.8,.1],[.9,.1],[.4,.6],[.6,.6],[.3,.9],[.3,.7],[.7,.3],[.5,.4]])
    session = Session(values,Config(m=1,terminal_solver='library'))
    selected,_ = session.recommend(method=method)
    for i in selected:
        for j in set(range(len(values)))-set(selected):
            assert not (np.all(values[j]>=values[i]) and np.any(values[j]>values[i]))


@pytest.mark.parametrize('method',['kl_dro','cvar','mmr','baseline_safe'])
def test_library_is_explicit_and_compared_with_complete_enumeration(method):
    full = Session(X,Config(m=1,terminal_solver='enumerate'))
    approximate = Session(X,Config(m=1,terminal_solver='library'))
    full.recommend(method=method)
    approximate.recommend(method=method)
    assert full.last_diagnostics['terminal_search_complete']
    assert not approximate.last_diagnostics['terminal_search_complete']
    assert approximate.last_diagnostics['terminal_risk'] >= full.last_diagnostics['terminal_risk']-1e-12
    if method == 'baseline_safe':
        assert approximate.last_diagnostics['constraint_violation_mass'] <= .1


def test_milp_fallback_is_feasible_and_fully_disclosed(monkeypatch):
    fake = {'solver_status':1,'solver_gap':.4,'solver_message':'test time limit',
            'terminal_search_complete':False,'terminal_scope':'full_pool_milp'}
    monkeypatch.setattr(adapter,'solve_milp',lambda *args,**kwargs:(None,fake.copy()))
    session = Session(X,Config(m=1,terminal_solver='milp'))
    session.recommend(method='baseline_safe')
    diagnostics = session.last_diagnostics
    assert not diagnostics['terminal_search_complete']
    assert diagnostics['fallback_from_milp']['solver_status'] == 1
    assert diagnostics['fallback_from_milp']['solver_gap'] == .4
    assert diagnostics['constraint_violation_mass'] <= .1
    strict = Session(X,Config(m=1,terminal_solver='milp',milp_fallback=False))
    with pytest.raises(RuntimeError):
        strict.recommend(method='cvar')


def test_literal_chance_constraint_rejects_numerically_unsafe_milp(monkeypatch):
    from types import SimpleNamespace
    from . import decisions
    u = np.array([[1.,0.],[0.,1.]])
    w = np.array([.5,.5])
    fake = SimpleNamespace(success=True,status=0,message='fake optimum',
                           x=np.array([0.,1.,0.,0.]),mip_gap=0.,mip_dual_bound=-.5,mip_node_count=1)
    monkeypatch.setattr(decisions,'milp',lambda *args,**kwargs:fake)
    result,status = solve_milp(u,w,1,'baseline_safe',baseline=(0,),epsilon=0.,delta=0.)
    assert result is None
    assert status['rejected'] == 'literal chance constraint failed'


def test_baseline_is_exactly_feasible_at_zero_risk_tolerance():
    # Regression: batched GEMM and separate GEMV scores differed by one ulp,
    # previously rejecting the baseline itself in 9/72 hypotheses.
    x = np.random.default_rng(7310).uniform(0,1,(8,3))
    session = Session(x,Config(m=1,safe_epsilon=0.,safe_delta=0.))
    base = session.baseline()
    u,w = session.prior.utilities,session.prior.probabilities
    slates = list(combinations(range(8),5))
    batch = baseline_violation_masses(u,slates,5,w,base,0.)
    single = np.array([baseline_violation_masses(u,[s],5,w,base,0.)[0] for s in slates])
    assert np.array_equal(batch,single)
    assert batch[slates.index(base)] == 0.
    selected,_ = session.recommend(method='baseline_safe')
    assert baseline_violation_masses(u,[selected],5,w,base,0.)[0] == 0.
    assert session.last_diagnostics['constraint_violation_mass'] == 0.


@pytest.mark.parametrize('parameter,value',[('skip_probability',0.),('skip_probability',1.),
                                             ('cvar_alpha',1.),('kl_radius',-1.),
                                             ('version_error_budget',True),('m',0),('m',7)])
def test_invalid_config_rejected(parameter,value):
    with pytest.raises(ValueError):
        Config(**{parameter:value})


def test_feature_only_interface_and_duplicate_pair_contract():
    with pytest.raises(TypeError):
        Session(X,true_utility=np.ones(8))
    with pytest.raises(TypeError):
        Session(X,case_seed=17)
    values = np.vstack((X,X[0],X[-1]))
    session = Session(values,Config(m=1))
    represented = {(tuple(values[a]),tuple(values[b])) for a,b in session.pairs}
    assert len(represented) == len(session.pairs)
    with pytest.raises(ValueError):
        session.recommend(k=True)
    with pytest.raises(ValueError):
        session.recommend(k=0)
