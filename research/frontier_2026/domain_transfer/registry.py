"""Explicit versions of two controls with an observed dominance boundary guard.

Reuse: existing AGPL-3.0 guarded_topk and frozen legacy controls, read and
smoked in guard-reuse-smoke-v1.json before adaptation. No old source changes.
The query rule is unchanged; it is not newly exact KG for the repaired rule.
"""
import numpy as np

from .. import registry as original
from ..components.legacy_controls.adapter import Equal, LinearEVOI
from research.nonlinear_topk.core import topk
from research.pabbo_topk.core import guarded_topk

BASE_METHODS = {
    'mixed_kg_finer_quadrature': 'mixed_kg_finer_quadrature',
    'P15_easy': 'P15_easy',
    'linear_evoi_3_boundary_guard': 'linear_evoi_3',
    'gp_topk_guard': 'gp_topk_guard',
    'pabbo_rl': 'pabbo_rl',
    'equal_boundary_guard': 'equal',
}
METHODS = tuple(BASE_METHODS)


def has_excluded_dominator(x, selected):
    selected = tuple(selected)
    outside = np.array([i for i in range(len(x)) if i not in selected], dtype=int)
    if not len(outside):
        return False
    a, b = x[list(selected)][None, :, :], x[outside][:, None, :]
    return bool(np.any(np.all(b >= a, axis=2) & np.any(b > a, axis=2)))


def repair(x, scores, selected):
    """Preserve any already valid frozen slate exactly, including its order."""
    if not has_excluded_dominator(x, selected):
        return tuple(selected), False
    order = topk(scores, len(x))
    priorities = np.empty(len(x), dtype=float)
    priorities[list(order)] = np.arange(len(x), 0, -1)
    # Integer priorities preserve the full frozen ranking without introducing
    # guarded_topk's six-decimal utility tie-break into unaffected cases.
    return guarded_topk(x, priorities, len(selected)), True


class BoundaryEqual(Equal):
    def recommend(self, history=(), *, acquisition='default', k=5):
        selected, query = super().recommend(history, acquisition=acquisition, k=k)
        selected, applied = repair(self.values, self.values.mean(axis=1), selected)
        self.last_diagnostics.update(boundary_dominance_guard_applied=applied,
                                     preserved_query_rule=True)
        return selected, query


class BoundaryLinear(LinearEVOI):
    def recommend(self, history=(), *, acquisition='default', k=5):
        selected, query = super().recommend(history, acquisition=acquisition, k=k)
        applied = has_excluded_dominator(self.values, selected)
        if applied:
            selected, _ = repair(self.values, self.belief(history).scores, selected)
        self.last_diagnostics.update(boundary_dominance_guard_applied=applied,
                                     preserved_query_rule=True)
        return selected, query


def description(name):
    if name not in BASE_METHODS:
        raise ValueError('Unknown domain-transfer method')
    base = BASE_METHODS[name]
    if base == name:
        return original.description(name)
    return {'component': 'domain_boundary_guard_v1', 'base_method': base,
            'base_description': original.description(base),
            'terminal_rule': 'preserve valid frozen slate; otherwise priority-topological repair',
            'ranking': 'unchanged frozen full rounded-score order',
            'query_rule': 'unchanged', 'strict_dominance_tolerance': 0,
            'claim': 'observed-feature contract repair, not a quality improvement claim'}


def create(name, values):
    if name == 'equal_boundary_guard':
        session = BoundaryEqual(values)
    elif name == 'linear_evoi_3_boundary_guard':
        session = BoundaryLinear(values)
    elif name in METHODS:
        return original.create(name, values)
    else:
        raise ValueError('Unknown domain-transfer method')
    return session, session.recommend
