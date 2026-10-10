"""Feature/answer-only sessions for the isolated risk-decision prototypes."""
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from research.frontier_2026.finite import FiniteModel, tradeoff_pairs
from research.frontier_2026.models import PRIOR_SEED, standard_prior
from .decisions import (TerminalResult, baseline_violation_masses, edge_gains, evaluate_risk, slate_losses,
                        solve_milp, solve_slates, top_slate, violation_mass)


@dataclass(frozen=True)
class Config:
    prior: str = "mixed"
    m: int = 4
    seed: int = PRIOR_SEED
    temperature: float = .04
    lapse: float = .02
    skip_probability: float = .15
    terminal_solver: str = "auto"
    enumeration_limit: int = 10_000
    milp_time_limit: float = 5.
    milp_fallback: bool = True
    kl_radius: float = .1
    cvar_alpha: float = .8
    version_error_budget: int = 1
    version_tie_margin: float = 0.
    mmr_query_limit: int = 12
    safe_epsilon: float = .01
    safe_delta: float = .1
    decision_tolerance: float = 1e-12

    def __post_init__(self):
        if self.prior not in {"mixed", "linear"}:
            raise ValueError("prior must be mixed or linear")
        for name, value, low, high in [
                ("m", self.m, 1, 6), ("seed", self.seed, 0, 2**63-1),
                ("enumeration_limit", self.enumeration_limit, 1, 100_000),
                ("version_error_budget", self.version_error_budget, 0, 3),
                ("mmr_query_limit", self.mmr_query_limit, 1, 5000)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if self.terminal_solver not in {"auto", "enumerate", "library", "milp"}:
            raise ValueError("unknown terminal solver")
        for name in ("temperature", "lapse", "skip_probability", "milp_time_limit",
                     "kl_radius", "cvar_alpha", "version_tie_margin", "safe_epsilon",
                     "safe_delta", "decision_tolerance"):
            if not np.isscalar(getattr(self, name)) or not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite scalar")
        if not (self.temperature > 0 and 0 <= self.lapse < 1
                and 0 < self.skip_probability < 1 and self.milp_time_limit > 0
                and self.kl_radius >= 0 and 0 <= self.cvar_alpha < 1
                and self.version_tie_margin >= 0 and self.safe_epsilon >= 0
                and 0 <= self.safe_delta <= 1 and self.decision_tolerance >= 0):
            raise ValueError("invalid probability, risk or tolerance parameter")
        if not isinstance(self.milp_fallback, bool):
            raise ValueError("milp_fallback must be boolean")


METHODS = {"kl_dro", "cvar", "mmr", "baseline_safe", "edgecut"}
VARIANTS = {
    "kl_dro": "P08_kl_dro_nominal_query",
    "cvar": "P09_cvar_nominal_query",
    "mmr": "P10_bounded_error_mmr_shortlist_query",
    "baseline_safe": "P11_baseline_safe_nominal_query",
    "edgecut": "P17_soft_regret_edgecut",
}


class VersionSpaceEmpty(ValueError):
    """Observed hard disagreements exceed every model's declared error budget."""


class Session:
    """Receives only normalized verified features and observed responses.

    No simulator case, true utility, family label or case seed is accepted.
    `Config.seed` is a declared fixed quadrature seed, not a simulator seed.
    All questions are global indices into canonical `pairs`.
    """
    def __init__(self, values, config=Config()):
        self.values = np.array(values, dtype=float, copy=True)
        self.pairs = tradeoff_pairs(self.values)
        self.values.setflags(write=False)
        if not isinstance(config, Config):
            raise TypeError("config must be risk_decisions.Config")
        self.config = config
        self.prior = standard_prior(self.values, kind=config.prior, m=config.m, seed=config.seed)
        self._models = {}
        self._enumerations = {}
        self._oracle_incidence = {}
        self.last_diagnostics = {}

    def _model(self, k):
        if k not in self._models:
            c = self.config
            self._models[k] = FiniteModel.pairwise(
                self.prior.utilities, self.pairs, k, prior=self.prior.probabilities,
                temperature=c.temperature, lapse=c.lapse, skip_probability=c.skip_probability)
        return self._models[k]

    def baseline(self, k=5):
        k = self._k(k)
        return top_slate(self.values.mean(axis=1), k, self.values.sum(axis=1))

    def _k(self, k):
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        return min(int(k), len(self.values))

    def _oracle_actions(self, k):
        tie = self.values.sum(axis=1)
        return tuple(top_slate(row, k, tie) for row in self.prior.utilities)

    def _library(self, w, k, extra=()):
        base = self.baseline(k)
        mean = top_slate(w @ self.prior.utilities, k, self.values.sum(axis=1))
        actions = set(self._oracle_actions(k)) | {base, mean}
        actions.update(tuple(sorted(s)) for s in extra)
        # All one-swap neighbors of two declared seeds, not an unbounded local
        # search that changes its budget after seeing evaluation performance.
        for seed in {base, mean}:
            outside = set(range(len(self.values)))-set(seed)
            for old in seed:
                for new in outside:
                    actions.add(tuple(sorted((set(seed)-{old}) | {new})))
        return tuple(sorted(actions))

    def _complete_actions(self, k):
        if comb(len(self.values), k) > self.config.enumeration_limit:
            raise ValueError("complete slate enumeration exceeds the declared limit")
        if k not in self._enumerations:
            self._enumerations[k] = tuple(combinations(range(len(self.values)), k))
        return self._enumerations[k]

    def _repair_dominance(self, selected):
        chosen = set(selected)
        # A replacement increases total observed skill, so finite iteration
        # cannot cycle. All default latent utilities are coordinatewise monotone.
        while True:
            replaced = False
            for i in sorted(chosen):
                dominators = [j for j in range(len(self.values)) if j not in chosen
                              and np.all(self.values[j] >= self.values[i])
                              and np.any(self.values[j] > self.values[i])]
                if dominators:
                    j = min(dominators, key=lambda j: (-self.values[j].sum(), j))
                    chosen.remove(i)
                    chosen.add(j)
                    replaced = True
                    break
            if not replaced:
                return tuple(sorted(chosen))

    def _version_counts(self, history, k):
        model = self._model(k)
        history = model.validate_history(history)
        errors = np.zeros(len(model.utilities), dtype=int)
        for q, r in history:
            if r == 2:
                continue
            a, b = self.pairs[q]
            difference = model.utilities[:, a]-model.utilities[:, b]
            errors += (difference < -self.config.version_tie_margin if r == 0
                       else difference > self.config.version_tie_margin)
        return errors

    def version_space(self, history=(), *, k=5):
        active = self._version_counts(history, self._k(k)) <= self.config.version_error_budget
        if not active.any():
            raise VersionSpaceEmpty("no utility hypothesis respects the declared response-error budget")
        return active

    def _terminal(self, w, k, method, active):
        c = self.config
        u = self.prior.utilities
        kwargs = dict(alpha=c.cvar_alpha, active=active, baseline=self.baseline(k),
                      epsilon=c.safe_epsilon, delta=c.safe_delta)
        mode = c.terminal_solver
        if mode == "auto":
            mode = ("enumerate" if comb(len(self.values), k) <= c.enumeration_limit
                    else "library" if method == "kl_dro" else "milp")
        fallback = None
        if mode == "milp":
            if method == "kl_dro":
                raise ValueError("the KL terminal has no MILP implementation")
            result, status = solve_milp(u, w, k, method, time_limit=c.milp_time_limit, **kwargs)
            if result is not None:
                return result
            if not c.milp_fallback:
                raise RuntimeError(f"MILP did not certify a feasible optimum: {status}")
            fallback = status
            mode = "library"
        actions = self._complete_actions(k) if mode == "enumerate" else self._library(w, k)
        result = solve_slates(u, w, k, method, actions, radius=c.kl_radius,
                              complete=mode == "enumerate", **kwargs)
        if fallback is not None:
            result.diagnostics["fallback_from_milp"] = fallback
            result.diagnostics["fallback_reason"] = fallback.get("rejected", fallback["solver_message"])
            result.diagnostics["terminal_search_complete"] = False
        return result

    def _nominal_query(self, model, w, available):
        # Deliberate implementation gap from P08/P09/P11 full risk acquisition:
        # only terminal decisions use risk; questions maximize nominal top-k VOI.
        gains = (model.one_step_values(w, available)-model.terminal(w))/model.k
        best = int(np.argmax(gains))
        value = float(gains[best])
        self.last_diagnostics.update(query_rule="nominal_one_step_topk_KG",
                                     query_gain=value, full_risk_acquisition=False)
        return available[best] if value > self.config.decision_tolerance else None

    def _mmr_query(self, history, model, w, active, selected, available, current):
        c = self.config
        errors = self._version_counts(history, model.k)
        branches = {}
        rank = []
        for q in available:
            a, b = self.pairs[q]
            diff = model.utilities[:, a]-model.utilities[:, b]
            left = errors+(diff < -c.version_tie_margin) <= c.version_error_budget
            right = errors+(diff > c.version_tie_margin) <= c.version_error_budget
            branches[q] = (left, right)
            counts = int(left.sum()), int(right.sum())
            strict = (int(np.count_nonzero(active & (diff < -c.version_tie_margin))),
                      int(np.count_nonzero(active & (diff > c.version_tie_margin))))
            rank.append((max(counts), sum(counts), max(strict), -min(strict), q))
        shortlist = [entry[-1] for entry in sorted(rank)[:c.mmr_query_limit]]
        complete = comb(len(self.values), model.k) <= c.enumeration_limit
        actions = (self._complete_actions(model.k) if complete
                   else self._library(w, model.k, extra=[selected]))
        losses = slate_losses(model.utilities, actions, model.k)
        options = []
        for q in shortlist:
            risks = [float(losses[:, branch].max(axis=1).min())
                     for branch in branches[q] if branch.any()]
            # Skip preserves the version space and consumes a slot. Its worst
            # branch is the current MMR, so use a disclosed two-answer tie rule.
            average = float(np.mean(risks))
            counts = [int(branch.sum()) for branch in branches[q]]
            a, b = self.pairs[q]
            diff = model.utilities[:, a]-model.utilities[:, b]
            strict = (int(np.count_nonzero(active & (diff < -c.version_tie_margin))),
                      int(np.count_nonzero(active & (diff > c.version_tie_margin))))
            options.append((average, max(counts), sum(counts), max(strict), -min(strict), q))
        best = min(options)
        self.last_diagnostics.update(
            query_rule="worst_branch_with_neutral_skip_then_uniform_binary_risk_tie_break",
            query_search_complete=len(shortlist) == len(available),
            query_shortlist_count=len(shortlist), query_leaf_action_scope="complete" if complete else "library",
            query_worst_including_skip=float(current), conditional_binary_risk=best[0],
            query_gain=float(current-best[0]), no_predictive_probabilities=True)
        # Balanced elimination is a final declared tie breaker to permit future
        # value when the immediate MMR is flat. It is not Bayesian information.
        best_q = best[-1]
        a, b = self.pairs[best_q]
        diff = model.utilities[:, a]-model.utilities[:, b]
        # With a positive error budget the first response may eliminate no
        # hypotheses, yet it changes their residual budgets. Do not stop merely
        # because all currently admissible sets are still equal.
        informative = (any(int(branch.sum()) < int(active.sum()) for branch in branches[best_q])
                       or (np.any(active & (diff < -c.version_tie_margin))
                           and np.any(active & (diff > c.version_tie_margin))))
        return best_q if (current-best[0] > c.decision_tolerance or informative) else None

    def recommend(self, history=(), *, method="kl_dro", k=5):
        if method not in METHODS:
            raise ValueError("unknown risk method")
        k = self._k(k)
        model = self._model(k)
        history = model.validate_history(history)
        active = self.version_space(history, k=k) if method == "mmr" else np.ones(len(model.utilities), dtype=bool)
        w = active/active.sum() if method == "mmr" else model.posterior(history)
        self.last_diagnostics = {"method": method, "variant": VARIANTS[method], "k": k,
                                 "response_slots_used": len(history), "hypotheses": len(w),
                                 "active_hypotheses": int(active.sum()), "global_query_indices": True,
                                 "belief": "bounded_error_version_space" if method == "mmr" else "finite_Bayes"}
        if method == "edgecut":
            selected = top_slate(w @ model.utilities, k, self.values.sum(axis=1))
            self.last_diagnostics.update(terminal_scope="posterior_mean_topk",
                                         terminal_search_complete=True, solver_status="sorting",
                                         full_risk_acquisition=True, floating_point=True)
        else:
            result = self._terminal(w, k, method, active)
            selected = result.selection
            self.last_diagnostics.update(result.diagnostics)
        repaired = self._repair_dominance(selected)
        self.last_diagnostics["dominance_replacements"] = len(set(repaired)-set(selected))
        selected = repaired
        loss = slate_losses(model.utilities, [selected], k)[0]
        self.last_diagnostics["nominal_expected_regret"] = float(loss @ w)
        risk = (float(loss @ w) if method == "edgecut" else evaluate_risk(
            loss, w, method, radius=self.config.kl_radius, alpha=self.config.cvar_alpha, active=active))
        self.last_diagnostics["terminal_risk"] = risk
        if method == "baseline_safe":
            bad = float(baseline_violation_masses(model.utilities, [selected], k, w,
                                                 self.baseline(k), self.config.safe_epsilon)[0])
            if bad > self.config.safe_delta:
                # A boundary mismatch after solver normalization/repair cannot
                # make the known feasible baseline disappear. Disclose fallback.
                selected = self.baseline(k)
                self.last_diagnostics.update(chance_postcheck_fallback={
                    "rejected_violation_mass": bad, "action": "fixed_baseline"},
                    terminal_search_complete=False)
                loss = slate_losses(model.utilities, [selected], k)[0]
                risk = float(loss @ w)
                self.last_diagnostics.update(nominal_expected_regret=risk, terminal_risk=risk)
                bad = 0.
            self.last_diagnostics.update(baseline=list(self.baseline(k)), constraint_violation_mass=bad,
                                         constraint_delta=self.config.safe_delta,
                                         constraint_is_population_guarantee=False)
        available = tuple(q for q in range(len(self.pairs)) if q not in {asked for asked, _ in history})
        if len(history) == 3 or k == len(self.values) or not available:
            self.last_diagnostics["query_rule"] = "stop_budget_or_single_action_or_no_questions"
            return selected, None
        if method == "mmr":
            query = self._mmr_query(history, model, w, active, selected, available, risk)
        elif method == "edgecut":
            if k not in self._oracle_incidence:
                z = np.zeros_like(model.utilities)
                z[np.arange(len(z))[:, None], np.array(self._oracle_actions(k))] = 1.
                self._oracle_incidence[k] = z
            gains = edge_gains(model.utilities, w, model.likelihood[list(available)], self._oracle_incidence[k], k)
            best = int(np.argmax(gains))
            self.last_diagnostics.update(query_rule="soft_regret_edge_expected_decrease",
                                         query_gain=float(gains[best]), ec2_theorem_applied=False)
            query = available[best] if gains[best] > self.config.decision_tolerance else None
        else:
            query = self._nominal_query(model, w, available)
        return selected, query
