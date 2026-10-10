"""Full one-step risk acquisitions on explicitly capped tiny finite problems.

These variants implement the P08/P09/P11 proposal's receding one-step criterion.
They do NOT solve a dynamically consistent risk problem or a three-step optimal
policy. All slates and all available queries are enumerated in floating point.
No registry entry or full-pool nominal-query variant is changed by this module.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from research.frontier_2026.finite import FiniteModel
from .adapter import Config, Session as NominalSession
from .decisions import baseline_violation_masses, evaluate_risk, slate_losses, solve_slates


MAX_N, MAX_H, MAX_Q = 8, 256, 28
VARIANTS = {"kl_dro": "P08_kl_dro_full_risk_query_tiny",
            "cvar": "P09_cvar_full_risk_query_tiny",
            "baseline_safe": "P11_baseline_safe_full_risk_query_tiny"}


@dataclass(frozen=True)
class RiskPlan:
    selection: tuple[int, ...]
    query: int | None
    current_risk: float
    value_with_stop: float
    diagnostics: dict


def plan_one_step(model: FiniteModel, history=(), *, method="kl_dro", radius=.1,
                  alpha=.8, baseline=None, epsilon=.01, delta=.1,
                  query_cost=0., tolerance=1e-12):
    """R(w)-sum_r p(r|w,q)R(w_r), with stop and a cost in regret units.

The posterior, including each hypothetical branch, uses FiniteModel.posterior
on the actual extended history. Thus constant neutral skips and impossible
histories follow exactly the same rule in planning and in the next real call.
P11 uses the SAME fixed baseline on every branch.
"""
    if method not in VARIANTS:
        raise ValueError("tiny full query supports KL-DRO, CVaR and baseline-safe only")
    if model.utilities.shape[1] > MAX_N or len(model.prior) > MAX_H or len(model.likelihood) > MAX_Q:
        raise ValueError(f"tiny scope cap is N<={MAX_N}, H<={MAX_H}, Q<={MAX_Q}")
    if model.likelihood.shape[1] != 3:
        raise ValueError("the FSP tiny prototype requires left/right/skip outcomes")
    if (not np.isscalar(query_cost) or not np.isfinite(query_cost) or query_cost < 0
            or not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("query_cost and tolerance must be nonnegative finite scalars")
    history = model.validate_history(history)
    weights = model.posterior(history)
    actions = tuple(combinations(range(model.utilities.shape[1]), model.k))

    def terminal(w):
        return solve_slates(model.utilities, w, model.k, method, actions,
                            radius=radius, alpha=alpha, baseline=baseline,
                            epsilon=epsilon, delta=delta, complete=True)

    current = terminal(weights)
    diagnostics = {**current.diagnostics, "method_variant": VARIANTS[method],
        "full_risk_acquisition": True, "horizon": 1, "global_three_step_optimal": False,
        "query_search_complete": True, "floating_point": True,
        "slots_used": len(history), "query_cost_normalized_regret": float(query_cost),
        "decision_tolerance": float(tolerance), "terminal_calls": 1,
        "query_records": [], "current_risk": current.risk,
        "tiny_caps": {"n": MAX_N, "h": MAX_H, "q": MAX_Q}}
    available = tuple(q for q in range(len(model.likelihood)) if q not in {q for q, _ in history})
    if len(history) == 3 or model.k == model.utilities.shape[1] or not available:
        diagnostics.update(stop_reason="no_remaining_decision", chosen_query=None,
                           value_with_stop=current.risk)
        return RiskPlan(current.selection, None, current.risk, current.risk, diagnostics)

    best_query, best_value = None, current.risk
    # Reuse identical distributions, including the exactly neutral skip. This
    # is only memoization of exact bytes, not rounding or particle compression.
    memo = {weights.tobytes(): current}
    for q in available:
        branches, future = [], 0.
        for outcome in range(3):
            probability = float(weights @ model.likelihood[q, outcome])
            if probability == 0.:
                branches.append({"outcome": outcome, "probability": 0.,
                                 "status": "impossible_no_posterior", "risk": None})
                continue
            posterior = model.posterior(history+((q, outcome),))
            key = posterior.tobytes()
            if key not in memo:
                memo[key] = terminal(posterior)
                diagnostics["terminal_calls"] += 1
            child = memo[key]
            future += probability*child.risk
            branches.append({"outcome": outcome, "probability": probability,
                             "status": "complete_slate_enumeration", "risk": child.risk,
                             "selection": child.selection})
        value = future+query_cost
        diagnostics["query_records"].append({"query": q, "future_risk": float(future),
            "gain_before_cost": float(current.risk-future), "risk_with_cost": float(value),
            "branches": branches})
        # Query id is the deterministic tie rule; tolerance only decides
        # whether an improvement justifies a question, never alters the risk.
        if value < best_value-tolerance:
            best_query, best_value = q, float(value)
    diagnostics.update(chosen_query=best_query, value_with_stop=best_value,
                       stop_reason="cost_or_no_positive_gain" if best_query is None else None)
    return RiskPlan(current.selection, best_query, current.risk, best_value, diagnostics)


class Session(NominalSession):
    """Feature-only adapter, N<=8 and m<=2; defaults to H=144 at d=3.

Import from this module explicitly. The default package Session remains the
larger-pool nominal-query implementation and retains its honest variant names.
"""
    def __init__(self, values, config=Config(m=2, terminal_solver="enumerate"), *, query_cost=0.):
        if not isinstance(config, Config) or config.m > 2 or config.terminal_solver != "enumerate":
            raise ValueError("tiny adapter requires risk Config(m<=2, terminal_solver='enumerate')")
        if len(values) > MAX_N:
            raise ValueError(f"tiny adapter supports at most {MAX_N} profiles")
        if not np.isscalar(query_cost) or not np.isfinite(query_cost) or query_cost < 0:
            raise ValueError("query_cost must be a nonnegative finite scalar")
        super().__init__(values, config)
        self.query_cost = float(query_cost)

    def recommend(self, history=(), *, method="kl_dro", k=5):
        k = self._k(k)
        model = self._model(k)
        c = self.config
        plan = plan_one_step(model, history, method=method, radius=c.kl_radius,
            alpha=c.cvar_alpha, baseline=self.baseline(k), epsilon=c.safe_epsilon,
            delta=c.safe_delta, query_cost=self.query_cost, tolerance=c.decision_tolerance)
        selected = self._repair_dominance(plan.selection)
        weights = model.posterior(history)
        loss = slate_losses(model.utilities, [selected], k)[0]
        actual_risk = evaluate_risk(loss, weights, method, radius=c.kl_radius,
                                   alpha=c.cvar_alpha, active=None)
        self.last_diagnostics = {**plan.diagnostics,
            "dominance_repair_changed_slate": selected != plan.selection,
            "returned_risk": actual_risk, "hypotheses": len(weights)}
        if actual_risk > plan.current_risk+1e-10:
            raise ArithmeticError("dominance closure unexpectedly increased the terminal risk")
        if method == "baseline_safe":
            bad = float(baseline_violation_masses(model.utilities, [selected], k, weights,
                                                  self.baseline(k), c.safe_epsilon)[0])
            self.last_diagnostics["returned_violation_mass"] = bad
            if bad > c.safe_delta:
                raise ArithmeticError("tiny full terminal violated its literal chance constraint")
        return selected, plan.query
