"""Contract and representation checks, not new quality/heldout experiments."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.preference_topk.core import Belief
from research.nonlinear_topk.core import GaussianBelief, posterior, recommend as gp_recommend
from research.pabbo_topk.core import recommend as neural_recommend
from ...finite import tradeoff_pairs
from .adapter import (CHECKPOINTS, FROZEN_GP, LINEAR_M, LINEAR_SEED, METHODS,
                      ROOT, Equal, GaussianTopK, LinearEVOI, PABBO,
                      _neural_model, make_session)


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def values(n=18, d=3):
    return np.random.default_rng(2026100891).uniform(.25, .95, (n, d))


def semantic_selection(x, selection):
    return [tuple(x[i]) for i in selection]


def test_frozen_parameters_and_sources_are_unchanged():
    smoke = json.loads((Path(__file__).parent/"source-smoke.json").read_text())
    for name, expected in smoke["source_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == expected
    protocol = json.loads((ROOT/"output/validation/nonlinear-topk-20261008/protocol.json").read_text())
    assert asdict(FROZEN_GP) == protocol["selected_config"]
    assert (LINEAR_M, LINEAR_SEED) == (9, 202610072)
    for path, expected in CHECKPOINTS.values():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("method", METHODS)
def test_coldstart_and_canonical_pool(method):
    x = values()
    session = make_session(method, x)
    assert session.pairs == tradeoff_pairs(x)
    selected, q = session.recommend()
    assert len(selected) == len(set(selected)) == 5
    assert all(type(i) is int and 0 <= i < len(x) for i in selected)
    assert q is None if method == "equal" else q in range(len(session.pairs))
    assert session.recommend() == (selected, q)
    assert not session.values.flags.writeable


@pytest.mark.parametrize("method", METHODS[1:])
def test_three_slots_and_neutral_skip(method):
    session = make_session(method, values())
    initial = session.recommend()[0]
    history = ()
    for _ in range(3):
        selected, q = session.recommend(history)
        assert selected == initial
        assert q is not None and q not in {old for old, _ in history}
        history += ((q, 2),)
    selected, q = session.recommend(history)
    assert selected == initial and q is None
    with pytest.raises(ValueError):
        session.recommend(history+((0, 0),))


@pytest.mark.parametrize("method", METHODS)
def test_mixed_history_mapping_no_repeats_and_budget(method):
    session = make_session(method, values())
    history = ((0, 0), (1, 2), (2, 1))
    selected, q = session.recommend(history)
    assert len(selected) == 5 and q is None
    _, q = session.recommend(history[:2])
    assert q not in {0, 1}
    with pytest.raises(ValueError):
        session.recommend(((0, 0), (0, 2)))


@pytest.mark.parametrize("method", METHODS)
def test_candidate_permutation_uses_same_vector_pairs_and_answers(method):
    x = values()
    order = np.random.default_rng(2026100892).permutation(len(x))
    first, moved = make_session(method, x), make_session(method, x[order])
    assert [(tuple(x[a]), tuple(x[b])) for a, b in first.pairs] == [
        (tuple(x[order][a]), tuple(x[order][b])) for a, b in moved.pairs]
    for history in [(), ((1, 0),), ((1, 0), (7, 2), (11, 1))]:
        a, qa = first.recommend(history)
        b, qb = moved.recommend(history)
        assert semantic_selection(x, a) == semantic_selection(x[order], b)
        assert qa == qb


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_dimensions_and_no_questions_for_one_monotone_skill(method, d):
    x = values(d=d)
    session = make_session(method, x)
    selected, q = session.recommend()
    assert len(selected) == 5
    if d == 1:
        assert q is None and session.pairs == ()
        assert set(selected) == set(np.argsort(-x[:, 0])[:5])
    else:
        assert q is None if method == "equal" else q in range(len(session.pairs))


@pytest.mark.parametrize("method", METHODS)
def test_duplicate_vectors_are_one_question_and_small_pool_stops(method):
    x = values(n=8)
    x = np.concatenate([x, x[[1, 3]]])
    session = make_session(method, x)
    keys = [(tuple(x[a]), tuple(x[b])) for a, b in session.pairs]
    assert len(keys) == len(set(keys))
    assert all(a != b for a, b in keys)
    small = make_session(method, x[:3])
    selected, q = small.recommend()
    assert set(selected) == {0, 1, 2} and q is None


@pytest.mark.parametrize("method", METHODS)
def test_invalid_inputs_and_history_are_rejected(method):
    for x in [np.zeros((4, 4)), np.zeros((0, 3)), np.zeros((101, 3)),
              [[np.nan, .2]], [[1.1, .2]]]:
        with pytest.raises(ValueError):
            make_session(method, x)
    session = make_session(method, values())
    for history in [((0, None),), ((0, True),), ((0, -1),), ((-1, 0),),
                    ((len(session.pairs), 0),), ((0, .5),), ((0, 3),)]:
        with pytest.raises(ValueError):
            session.recommend(history)
    for k in [0, True, 6, 1.5]:
        with pytest.raises(ValueError):
            session.recommend(k=k)
    with pytest.raises(ValueError):
        session.recommend(acquisition="hidden_oracle")


def test_answer_orientation_matches_frozen_linear_and_gp_cores():
    x = values()
    linear, gp = LinearEVOI(x), GaussianTopK(x)
    q = next(q for q, (a, b) in enumerate(linear.pairs) if a > b)
    pair = linear.pairs[q]
    for answer in [0, 1, 2]:
        original_choice = None if answer == 2 else answer
        direct = Belief.linear(x).observe(pair, original_choice)
        np.testing.assert_array_equal(linear.belief(((q, answer),)).scores, direct.scores)
        direct_gp = posterior(x, [(pair, original_choice)], FROZEN_GP)
        np.testing.assert_array_equal(gp.belief(((q, answer),)).mean, direct_gp.mean)


@pytest.mark.parametrize("variant", ["published", "rl"])
def test_pabbo_vector_index_bridge_matches_existing_transfer(variant):
    x = values()
    session = PABBO(x, variant=variant)
    q = next(q for q, (a, b) in enumerate(session.pairs) if a > b)
    a, b = session.pairs[q]
    for answer in [0, 1, 2]:
        history = [{"left": x[a].tolist(), "right": x[b].tolist(),
                    "choice": None if answer == 2 else answer}]
        direct = neural_recommend(session._model, x, history)
        selected, next_q = session.recommend(((q, answer),))
        assert selected == tuple(direct["selected"])
        pair = session.pairs[next_q]
        assert tuple(x[pair[0]]) == tuple(direct["question"]["left"])
        assert tuple(x[pair[1]]) == tuple(direct["question"]["right"])


def test_dominance_guard_is_new_and_uses_only_observed_features(monkeypatch):
    x = np.array([[.2, .2], [.9, .9], [.8, .3], [.3, .8], [.6, .6], [.4, .5]])
    belief = GaussianBelief(np.array([10., 0., 9., 8., 7., 6.]), np.eye(len(x)))
    plain, guarded = GaussianTopK(x), GaussianTopK(x, guard=True)
    monkeypatch.setattr(plain, "belief", lambda history=(): belief)
    monkeypatch.setattr(guarded, "belief", lambda history=(): belief)
    a, qa = plain.recommend(k=2)
    b, qb = guarded.recommend(k=2)
    assert a == (0, 2) and b != a
    assert qa == qb  # Acquisition deliberately remains the original GP criterion.
    outside = set(range(len(x)))-set(b)
    assert all(not (np.all(x[j] >= x[i]) and np.any(x[j] > x[i]))
               for i in b for j in outside)


def test_gp_original_decision_and_high_gain_question_agree_on_fixture():
    x = values()
    direct, query, _ = gp_recommend(x, config=FROZEN_GP)
    session = GaussianTopK(x)
    selected, q = session.recommend()
    assert selected == direct
    assert set(session.pairs[q]) == set(query.pair)


def test_models_and_calls_preserve_external_rng_and_default_dtype():
    _neural_model.cache_clear()
    torch_state = torch.get_rng_state().clone()
    np_state = np.random.get_state()
    dtype = torch.get_default_dtype()
    for method in METHODS:
        session = make_session(method, values())
        session.recommend(((1, 0),))
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_default_dtype() == dtype
    after = np.random.get_state()
    assert after[0] == np_state[0] and after[2:] == np_state[2:]
    np.testing.assert_array_equal(after[1], np_state[1])


@pytest.mark.parametrize("factory", [Equal, LinearEVOI, GaussianTopK])
def test_symmetric_non_neural_priors_under_feature_permutation(factory):
    x = values()
    original, permuted = factory(x), factory(x[:, [2, 0, 1]])
    assert set(original.recommend()[0]) == set(permuted.recommend()[0])
    # PABBO is intentionally absent: its learned coordinate embeddings have no
    # structural feature-permutation guarantee. Padding is preserved, not hidden.


@pytest.mark.parametrize("method", METHODS)
def test_hundred_candidate_contract(method):
    x = values(n=100)
    session = make_session(method, x)
    selected, q = session.recommend()
    assert len(selected) == 5 and len(set(selected)) == 5
    assert q is None if method == "equal" else q in range(len(session.pairs))


def test_frozen_equal_tie_rule_has_a_documented_permutation_limit():
    # Retaining legacy controls means not silently changing their terminal rule.
    # Different vectors with equal scores are chosen in input order by old topk.
    x = np.array([[.9, .1], [.1, .9], [.7, .3], [.3, .7], [.6, .4], [.4, .6]])
    selected = Equal(x).recommend()[0]
    reversed_x = x[::-1]
    moved = Equal(reversed_x).recommend()[0]
    assert semantic_selection(x, selected) != semantic_selection(reversed_x, moved)


@pytest.mark.parametrize("method", METHODS)
def test_none_acquisition_stops_without_changing_selection(method):
    session = make_session(method, values())
    for history in [(), ((0, 1),)]:
        normal_selection, _ = session.recommend(history)
        stopped_selection, question = session.recommend(history, acquisition="none")
        assert stopped_selection == normal_selection
        assert question is None
