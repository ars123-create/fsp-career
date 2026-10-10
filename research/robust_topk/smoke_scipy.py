"""Isolated reproduction of SciPy's documented MILP example.

Example attribution: SciPy developers, BSD-3-Clause, scipy.optimize.milp docs.
Run before adapting the solver to our own model.
"""
import json
import platform
import numpy as np
import scipy
from scipy.optimize import milp, LinearConstraint


def main():
    objective = -np.array([0.0, 1.0])
    matrix = np.array([[-1., 1.], [3., 2.], [2., 3.]])
    constraints = LinearConstraint(matrix, -np.inf, [1., 12., 12.])
    integer = milp(objective, integrality=np.ones(2), constraints=constraints)
    relaxed = milp(objective, constraints=constraints)
    assert integer.success and np.allclose(integer.x, [1, 2])
    assert relaxed.success and np.allclose(relaxed.x, [1.8, 2.8])
    print(json.dumps({"status": "PASS", "python": platform.python_version(),
                      "numpy": np.__version__, "scipy": scipy.__version__,
                      "integer": integer.x.tolist(), "relaxed": relaxed.x.tolist()}))


if __name__ == "__main__":
    main()
