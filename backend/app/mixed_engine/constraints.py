"""Observed-feature dominance closure for the research shortlist.

Extracted from the independently tested internal risk-decisions repair pattern.
Each replacement strictly increases a selected observed vector componentwise,
so a finite pool cannot cycle. This is a hard contract, not a utility learner.
"""
import numpy as np


def hypothesis_topk(values, utilities, k):
    """Stable utility sort with a dominance-compatible observed-feature tie rule.

    The rule matters for flat/floored hypotheses, not just the final mean.
    Candidate identity among identical observed vectors remains interchangeable.
    """
    x, utilities = np.asarray(values), np.asarray(utilities)
    canonical = np.array(sorted(range(len(x)), key=lambda i: (*(-x[i]), i)))
    positions = np.argsort(-utilities[:, canonical], axis=1, kind='stable')[:, :k]
    return canonical[positions]


def monotone_selection(values, scores, k):
    x, scores = np.asarray(values), np.asarray(scores)
    order = sorted(range(len(x)), key=lambda i: (-float(scores[i]), *(-x[i]), i))
    selected = set(order[:k])
    while True:
        change = None
        for i in sorted(selected):
            for j in order:
                if j not in selected and np.all(x[j] >= x[i]) and np.any(x[j] > x[i]):
                    change = i, j
                    break
            if change is not None:
                break
        if change is None:
            return tuple(i for i in order if i in selected)
        selected.remove(change[0])
        selected.add(change[1])
