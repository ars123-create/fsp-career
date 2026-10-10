"""Reproduce SciPy's public Sobol example before adapting the component.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
SciPy is BSD-3-Clause; the installed wheel license is retained in licenses/.
"""
import numpy as np
from scipy.stats import qmc


def main():
    sample = qmc.Sobol(d=2, scramble=False).random_base2(m=3)
    expected = np.array([[0, 0], [.5, .5], [.75, .25], [.25, .75],
                         [.375, .375], [.875, .875], [.625, .125], [.125, .625]])
    np.testing.assert_array_equal(sample, expected)
    np.testing.assert_allclose(qmc.discrepancy(sample), .013882107204860938)
    print("PASS: official Sobol 8-point example and discrepancy")


if __name__ == "__main__":
    main()
