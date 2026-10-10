import copy
import numpy as np
import pytest
import torch

from .adapter import Config, MoNNE, PairwiseRankingNet


def values():
    return np.array([[.9, .4, .8], [.5, .9, .7], [.6, .7, .9],
                     [.8, .6, .5], [.4, .8, .9], [.9, .5, .6]])


def test_actual_activation_equals_monotone_squared_sigmoid_value_and_gradient():
    torch.manual_seed(47)
    original = PairwiseRankingNet(3, True).double()
    explicit = copy.deepcopy(original)
    class SquaredSigmoid(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(x).square()
    explicit.af = SquaredSigmoid()
    explicit.af2 = SquaredSigmoid()
    x = torch.tensor(values(), dtype=torch.double)
    a, b = original(x, x)[0], explicit(x, x)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.sum().backward()
    b.sum().backward()
    for pa, pb in zip(original.parameters(), explicit.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-12, rtol=1e-12)


def test_cold_start_skip_no_duplicate_update_and_anchored_scale():
    torch.set_num_threads(2)
    session = MoNNE(values(), Config(ensemble_size=3))
    before = session.hypotheses()
    skipped = session.hypotheses(((0, 2),))
    assert before is skipped
    assert np.isfinite(before).all()
    assert np.min(before) >= 0 and np.max(before) <= 1
    selected, query = session.recommend()
    assert len(selected) == 5
    assert query in range(len(session.pairs)) or query is None
    trained = session.hypotheses(((0, 0),))
    assert np.isfinite(trained).all()
    assert not np.allclose(trained, before)
    with pytest.raises(ValueError):
        session.recommend(((0, 0), (0, 1)))


def test_duplicate_profiles_do_not_create_zero_scale():
    session = MoNNE(np.full((6, 3), .5), Config(ensemble_size=2))
    scores = session.hypotheses()
    assert np.isfinite(scores).all()
    assert np.max(scores-scores[:, :1]) == 0
    selected, query = session.recommend()
    assert selected == (0, 1, 2, 3, 4)
    assert query is None


def test_global_rng_and_dtype_restored():
    before_dtype = torch.get_default_dtype()
    before_rng = torch.get_rng_state().clone()
    session = MoNNE(values(), Config(ensemble_size=2))
    session.hypotheses(((0, 0),))
    assert torch.get_default_dtype() == before_dtype
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_monotonicity_under_increases_before_and_after_training():
    x = values()
    x = np.concatenate([x, np.minimum(x+.1, 1.)])
    session = MoNNE(x, Config(ensemble_size=2))
    for history in [(), ((0, 0),)]:
        u = session.hypotheses(history)
        assert np.min(u[:, 6:]-u[:, :6]) >= -1e-12


@pytest.mark.parametrize("acquisition", ["top5", "ieubo", "eubo", "information"])
def test_three_answer_budget_and_all_acquisitions(acquisition):
    session = MoNNE(values(), Config(ensemble_size=2))
    history = ((0, 0), (1, 2), (2, 1))
    selected, question = session.recommend(history, acquisition=acquisition)
    assert len(selected) == 5
    assert question is None
    _, question = session.recommend(((0, 2),), acquisition=acquisition)
    assert question != 0


def test_small_pool_stops_and_roundoff_ties_respect_dominance():
    for n in (1, 4, 5):
        x = np.linspace(.1, .9, n)[:, None]
        session = MoNNE(x, Config(ensemble_size=2))
        selected, query = session.recommend(acquisition='ieubo')
        assert len(selected) == n
        assert query is None
    x = 1e-13*np.array([[1,9],[3,7],[5,5],[7,3],[9,1],[9,9]])
    session = MoNNE(x, Config(ensemble_size=2))
    selected, _ = session.recommend()
    assert 5 in selected
