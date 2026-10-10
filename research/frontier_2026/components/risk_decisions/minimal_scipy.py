"""SciPy official MILP example, then a separate KL-constrained smoke problem.

MILP example adapted from scipy.optimize.milp documentation (BSD-3-Clause).
See SCIPY_LICENSE.txt and sources.md. Not a performance/quality experiment.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import LinearConstraint, milp, minimize
from scipy.special import rel_entr


def run():
    c = -np.array([0., 1.])
    a = np.array([[-1., 1.], [3., 2.], [2., 3.]])
    constraint = LinearConstraint(a, -np.inf, [1., 12., 12.])
    integer = milp(c, constraints=constraint, integrality=np.ones(2))
    relaxed = milp(c, constraints=constraint)
    assert integer.success and relaxed.success
    assert np.allclose(integer.x, [1., 2.])
    assert np.allclose(relaxed.x, [1.8, 2.8])
    p = np.array([.3, .7])
    loss = np.array([1., 0.])
    radius = .1
    result = minimize(lambda nu: -nu @ loss, p.copy(), method="SLSQP",
                      bounds=[(1e-12, 1.)]*2,
                      constraints=[{"type": "eq", "fun": lambda nu: nu.sum()-1.},
                                   {"type": "ineq", "fun": lambda nu: radius-rel_entr(nu, p).sum()}],
                      options={"ftol": 1e-12, "maxiter": 1000})
    assert result.success
    assert abs(result.x.sum()-1.) < 1e-10
    assert rel_entr(result.x, p).sum() <= radius+1e-9
    source = Path(inspect.getsourcefile(milp))
    return {"utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
            "scipy_version": scipy.__version__, "milp_source": str(source),
            "milp_source_sha256": sha256(source.read_bytes()).hexdigest(),
            "milp_integer": integer.x.tolist(), "milp_relaxation": relaxed.x.tolist(),
            "kl_smoke": {"weights": result.x.tolist(), "risk": float(result.x @ loss),
                         "kl": float(rel_entr(result.x, p).sum())}}


if __name__ == "__main__":
    result = run()
    output = Path(__file__).with_name("minimal-scipy-result.json")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
