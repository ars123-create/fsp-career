"""Reproduce small library operations before custom categorical adaptation."""
import hashlib
import inspect
import json
import platform
from pathlib import Path

import botorch
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.models import SingleTaskGP
import numpy as np
import scipy
import torch

from research.preference_topk.core import Belief, questions, simplex_particles


def main():
    out = Path(__file__).resolve().parent/'evidence/reuse-smoke.json'
    if out.exists():
        raise SystemExit('Refusing overwrite')
    torch.set_num_threads(2)
    torch.manual_seed(2026100801)
    x = torch.rand(8, 2, dtype=torch.double)
    y = (x[:, 0]+.5*x[:, 1]).unsqueeze(-1)
    model = SingleTaskGP(x, y)
    acq = qMultiStepLookahead(model, batch_sizes=[1, 1], num_fantasies=[2, 2])
    z = torch.rand(acq.get_augmented_q_batch_size(1), 2, dtype=torch.double)
    value = float(acq(z).detach().item())
    values = np.array([[.9, .4, .7], [.5, .9, .8], [.7, .7, .6], [.6, .6, .9]])
    belief = Belief.linear(values, m=4)
    qs = questions(belief, 2)
    record = {
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'torch': torch.__version__, 'botorch': botorch.__version__,
        'botorch_example': {'augmented_size': len(z), 'two_step_value': value,
                            'finite': bool(np.isfinite(value)),
                            'scope': 'Small continuous GP; not preference adaptation or optimized policy'},
        'existing_particle_model': {'shape': list(belief.utilities.shape),
                                    'prior_mass': float(belief.probabilities.sum()),
                                    'query_count': len(qs),
                                    'min_one_step_gain': min(q.expected_gain for q in qs)},
        'sources': {},
        'earlier_attempt': 'Computations completed but metadata capture failed: inspect.getfile on lru_cache wrapper; fixed with inspect.unwrap.',
    }
    for symbol in [qMultiStepLookahead, simplex_particles]:
        p = Path(inspect.getfile(inspect.unwrap(symbol)))
        record['sources'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
