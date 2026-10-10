"""Finite-scenario risk decisions; mathematical helpers, not hiring scores.

Original project implementation using SciPy's unmodified optimizers. The MILP
formulations and finite KL dual are documented in README.md and sources.md.
Complete enumeration/MILP describe the search scope, not exact real arithmetic.
"""
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, brentq, milp
from scipy.special import logsumexp


def probabilities(weights):
    w = np.asarray(weights, dtype=float)
    if (w.ndim != 1 or not len(w) or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("weights must be a finite probability vector")
    return w / w.sum()


def checked_table(utilities, weights, k):
    u = np.asarray(utilities, dtype=float)
    w = probabilities(weights)
    if (u.ndim != 2 or u.shape[0] != len(w) or u.shape[1] == 0
            or not np.isfinite(u).all()):
        raise ValueError("expected finite H by N utility table")
    if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 1 <= k <= u.shape[1]):
        raise ValueError("k must be an integer in 1..N")
    return u, w, int(k)


def top_slate(scores, k, tie_scores=None):
    scores = np.asarray(scores)
    tie = np.zeros(len(scores)) if tie_scores is None else np.asarray(tie_scores)
    order = np.lexsort((np.arange(len(scores)), -tie, -scores))
    return tuple(sorted(map(int, order[:k])))


def slate_losses(utilities, slates, k):
    """A by H normalized regret; every action has exactly k distinct indices."""
    u = np.asarray(utilities, dtype=float)
    slates = tuple(slates)
    if not slates:
        raise ValueError("slate set cannot be empty")
    for s in slates:
        if len(s) != k or len(set(s)) != k or any(i < 0 or i >= u.shape[1] for i in s):
            raise ValueError("invalid slate")
    incidence = np.zeros((len(slates), u.shape[1]))
    incidence[np.arange(len(slates))[:, None], np.array(slates)] = 1.
    oracle = np.sort(u, axis=1)[:, -k:].sum(axis=1)
    losses = (oracle[:, None] - u @ incidence.T).T / k
    # Roundoff can produce a negative zero-size regret at an oracle action.
    return np.maximum(losses, 0.)


def kl_worst_case(loss, weights, radius):
    """sup E_nu[loss], KL(nu || weights)<=radius, solved by exponential tilt.

    Returns a finite-support numerical solution. Zero nominal probabilities
    remain zero; this ambiguity set cannot invent out-of-support utilities.
    The tilt parameter solves the first-order condition of the entropic dual.
    """
    loss = np.asarray(loss, dtype=float)
    w = probabilities(weights)
    if loss.shape != w.shape or not np.isfinite(loss).all():
        raise ValueError("loss must match the probability vector")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("KL radius must be finite and nonnegative")
    support = w > 0
    p, x = w[support], loss[support]
    if radius == 0:
        return float(p @ x)
    largest = float(x.max())
    top_mass = float(p[x == largest].sum())
    if radius >= -np.log(top_mass):
        return largest
    spread = largest-float(x.min())
    if spread == 0:
        return largest
    # Scale to [-1,0] so bracketing is insensitive to cardinal utility units.
    v = (x-largest)/spread
    logp = np.log(p)

    def tilted(t):
        z = logp+t*v
        logz = logsumexp(z)
        nu = np.exp(z-logz)
        divergence = float(t*(nu @ v)-logz)
        return nu, divergence

    high = 1.
    while tilted(high)[1] < radius:
        high *= 2
        if not np.isfinite(high):
            raise ArithmeticError("KL tilt failed to bracket its dual optimum")
    t = brentq(lambda value: tilted(value)[1]-radius, 0., high,
               xtol=1e-12, rtol=1e-12, maxiter=200)
    nu, _ = tilted(t)
    return float(nu @ x)


def cvar(losses, weights, alpha):
    """Upper-tail CVaR with fractional atoms; alpha=0 is the expectation."""
    loss = np.asarray(losses, dtype=float)
    w = probabilities(weights)
    if (loss.ndim not in (1, 2) or loss.shape[-1] != len(w)
            or not np.isfinite(loss).all()):
        raise ValueError("losses must have hypotheses on the final axis")
    if not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("CVaR alpha must be in [0,1)")
    if alpha == 0:
        return loss @ w
    order = np.argsort(-loss, axis=-1, kind="stable")
    ordered = np.take_along_axis(loss, order, axis=-1)
    probability = w[order]
    prefix = np.cumsum(probability, axis=-1)-probability
    take = np.minimum(probability, np.maximum(1-alpha-prefix, 0.))
    return (take*ordered).sum(axis=-1)/(1-alpha)


def violation_mass(losses, weights, baseline_loss, epsilon):
    """Literal finite-hypothesis constraint, with a strict loss comparison."""
    return (np.asarray(losses) > np.asarray(baseline_loss)+epsilon) @ weights


def baseline_violation_masses(u, slates, k, weights, baseline, epsilon):
    """The same chance constraint with common terms cancelled before arithmetic.

    L(S)-L(S0) = [sum_(S0\\S) U - sum_(S\\S0) U]/k. Computing this
    directly guarantees that S0 compared to itself has exactly zero difference,
    even at epsilon=delta=0. Batch and individual calls follow the same path.
    """
    base = set(baseline)
    result = []
    for slate in slates:
        chosen = set(slate)
        removed, added = sorted(base-chosen), sorted(chosen-base)
        difference = (u[:, removed].sum(axis=1)-u[:, added].sum(axis=1))/k
        result.append(float(weights[difference > epsilon].sum()))
    return np.array(result)


@dataclass(frozen=True)
class TerminalResult:
    selection: tuple[int, ...]
    risk: float
    diagnostics: dict


def evaluate_risk(loss, w, method, *, radius, alpha, active):
    if method == "kl_dro":
        return kl_worst_case(loss, w, radius)
    if method == "cvar":
        return float(cvar(loss, w, alpha))
    if method == "mmr":
        return float(np.max(loss[active]))
    if method == "baseline_safe":
        return float(loss @ w)
    raise ValueError("unknown terminal risk method")


def solve_slates(u, w, k, method, slates, *, radius=.1, alpha=.8,
                 active=None, baseline=None, epsilon=.01, delta=.1,
                 complete=False):
    u, w, k = checked_table(u, w, k)
    slates = tuple(sorted(set(tuple(sorted(s)) for s in slates)))
    losses = slate_losses(u, slates, k)
    active = np.ones(len(u), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if active.shape != (len(u),) or not active.any():
        raise ValueError("nonempty active hypothesis set required")
    expected = losses @ w
    eligible = np.ones(len(slates), dtype=bool)
    checked = len(slates)
    if method == "kl_dro":
        # E_w L is a lower bound on the KL supremum. It safely prunes actions
        # only after a better robust feasible action has actually been found.
        risks = np.full(len(slates), np.inf)
        checked, best = 0, np.inf
        for i in np.argsort(expected, kind="stable"):
            if expected[i] > best+1e-12:
                break
            risks[i] = kl_worst_case(losses[i], w, radius)
            best = min(best, risks[i])
            checked += 1
    elif method == "cvar":
        risks = cvar(losses, w, alpha)
    elif method == "mmr":
        risks = losses[:, active].max(axis=1)
    elif method == "baseline_safe":
        if baseline is None or tuple(sorted(baseline)) not in slates:
            raise ValueError("baseline must be in the feasible action library")
        eligible = baseline_violation_masses(u, slates, k, w, baseline, epsilon) <= delta
        risks = expected.copy()
        risks[~eligible] = np.inf
    else:
        raise ValueError("unknown terminal risk method")
    # Stable secondary mean regret; this does not perturb the primary objective.
    chosen = int(np.lexsort((np.arange(len(slates)), expected, risks))[0])
    if not np.isfinite(risks[chosen]):
        raise ArithmeticError("no feasible action, including the baseline")
    diagnostics = {"terminal_scope": "complete_slate_enumeration" if complete else "restricted_slate_library",
                   "terminal_search_complete": bool(complete), "solver_status": "enumerated",
                   "solver_gap": None, "slate_count": len(slates), "risk_evaluations": checked,
                   "feasible_slate_count": int(eligible.sum()), "floating_point": True}
    return TerminalResult(slates[chosen], float(risks[chosen]), diagnostics)


def solve_milp(u, w, k, method, *, alpha=.8, active=None, baseline=None,
               epsilon=.01, delta=.1, time_limit=5.):
    """Full-pool MILP for CVaR, MMR and the scenario chance constraint.

    A time limit/infeasibility does not count as an optimal terminal solution.
    The caller may explicitly record and choose a feasible-library fallback.
    """
    u, w, k = checked_table(u, w, k)
    h, n = u.shape
    oracle = np.sort(u, axis=1)[:, -k:].sum(axis=1)/k
    active = np.ones(h, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if active.shape != (h,) or not active.any():
        raise ValueError("nonempty active hypothesis set required")
    if method == "mmr":
        size = n+1
        cost = np.r_[np.zeros(n), 1.]
        lower, upper = np.r_[np.zeros(n), 0.], np.r_[np.ones(n), np.inf]
        integer = np.r_[np.ones(n), 0]
        a = np.column_stack((u[active]/k, np.ones(active.sum())))
        constraints = [LinearConstraint(a, oracle[active], np.inf)]
    elif method == "cvar":
        if not 0 <= alpha < 1:
            raise ValueError("CVaR alpha must be in [0,1)")
        size = n+1+h
        cost = np.r_[np.zeros(n), 1., w/(1-alpha)]
        lower = np.zeros(size)
        upper = np.r_[np.ones(n), np.full(1+h, np.inf)]
        integer = np.r_[np.ones(n), np.zeros(1+h)]
        a = np.column_stack((u/k, np.ones(h), np.eye(h)))
        constraints = [LinearConstraint(a, oracle, np.inf)]
    elif method == "baseline_safe":
        if baseline is None or len(baseline) != k or len(set(baseline)) != k:
            raise ValueError("valid baseline required")
        size = n+h
        cost = np.r_[-w @ u/k, np.zeros(h)]
        lower, upper, integer = np.zeros(size), np.ones(size), np.ones(size)
        base_utility = u[:, baseline].sum(axis=1)/k
        smallest = np.sort(u, axis=1)[:, :k].sum(axis=1)/k
        big_m = np.maximum(base_utility-smallest-epsilon, 0.)
        a = np.column_stack((u/k, np.diag(big_m)))
        constraints = [LinearConstraint(a, base_utility-epsilon, np.inf),
                       LinearConstraint(np.r_[np.zeros(n), w], -np.inf, delta)]
    else:
        raise ValueError("MILP supports cvar, mmr and baseline_safe only")
    constraints.append(LinearConstraint(np.r_[np.ones(n), np.zeros(size-n)], k, k))
    result = milp(cost, integrality=integer, bounds=Bounds(lower, upper),
                  constraints=constraints,
                  options={"time_limit": float(time_limit), "mip_rel_gap": 0., "presolve": True})
    diagnostic = {"terminal_scope": "full_pool_milp", "terminal_search_complete": bool(result.success),
                  "solver_status": int(result.status), "solver_message": str(result.message),
                  "solver_gap": None if result.mip_gap is None else float(result.mip_gap),
                  "solver_dual_bound": None if result.mip_dual_bound is None else float(result.mip_dual_bound),
                  "solver_nodes": None if result.mip_node_count is None else int(result.mip_node_count),
                  "floating_point": True}
    if not result.success or result.x is None:
        return None, diagnostic
    chosen = tuple(map(int, np.flatnonzero(result.x[:n] > .5)))
    if len(chosen) != k or np.max(np.abs(result.x[:n]-np.round(result.x[:n]))) > 1e-6:
        diagnostic.update(terminal_search_complete=False, rejected="invalid integer slate")
        return None, diagnostic
    loss = slate_losses(u, [chosen], k)[0]
    if method == "baseline_safe":
        bad = float(baseline_violation_masses(u, [chosen], k, w, baseline, epsilon)[0])
        diagnostic["constraint_violation_mass"] = bad
        if bad > delta:
            # The literal discrete constraint is stronger than a solver's
            # numerical feasibility tolerance; never silently return a violation.
            diagnostic.update(terminal_search_complete=False, rejected="literal chance constraint failed")
            return None, diagnostic
    risk = evaluate_risk(loss, w, method, radius=0., alpha=alpha, active=active)
    return TerminalResult(chosen, risk, diagnostic), diagnostic


def edge_disagreement(u, w, oracle_incidence, k):
    """Exact algebraic O(HN) form of the soft-regret H² edge sum."""
    oracle = np.sort(u, axis=1)[:, -k:].sum(axis=1)
    return float((w @ oracle - (w @ u) @ (w @ oracle_incidence))/(2*k))


def edge_gains(u, w, likelihood, oracle_incidence, k):
    """Normalized-posterior soft-edge acquisition, not the EC² algorithm."""
    current = edge_disagreement(u, w, oracle_incidence, k)
    mass = likelihood*w[None, None, :]
    probability = mass.sum(axis=2)
    oracle = np.sort(u, axis=1)[:, -k:].sum(axis=1)
    cross = np.sum((mass @ u)*(mass @ oracle_incidence), axis=2)
    continuation = (mass @ oracle - np.divide(cross, probability,
                                             out=np.zeros_like(cross), where=probability > 0))/(2*k)
    return current-continuation.sum(axis=1)
