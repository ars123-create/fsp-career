"""Official Hypothesis minimal example, before the item-generator adaptation.

Source: https://hypothesis.readthedocs.io/en/latest/tutorial/introduction.html
Hypothesis 6.168.5 / MPL-2.0; full license remains in licenses/hypothesis-0-LICENSE.txt.
"""
import json
import hypothesis
from hypothesis import given, strategies as st, find, settings


@given(st.integers())
def test_is_integer(n):
    assert isinstance(n, int)


if __name__ == "__main__":
    test_is_integer()
    witness = find(st.integers(0, 20), lambda n: n * n != n,
                   settings=settings(derandomize=True, database=None))
    assert witness == 2
    print(json.dumps({"status": "PASS", "hypothesis": hypothesis.__version__,
                      "official_integer_example": "PASS", "find_witness": witness}))
