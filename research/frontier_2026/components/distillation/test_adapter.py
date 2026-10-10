"""P20 engineering checks; no claim that dev metrics establish superiority."""
from dataclasses import replace
from hashlib import sha256
import inspect
import json

import numpy as np
import pytest

from research.frontier_2026.finite import FiniteModel, tradeoff_pairs
from research.frontier_2026.components.distillation.adapter import (
    Config, FEATURE_NAMES, HERE, MODES, Session, Student, action_values,
)
from research.frontier_2026.components.distillation.train import (
    TRAIN_SEEDS, AGGREGATION_SEEDS, DEV_SEEDS, TINY_SEEDS,
)


X = np.array([[.1,.9],[.3,.7],[.7,.3],[.9,.1],[.6,.6],[.2,.2],[.8,.8]])


@pytest.mark.parametrize("mode", sorted(MODES))
def test_modes_coldstart_three_slots(mode):
    s = Session(X, Config(mode=mode))
    history = ()
    for _ in range(3):
        selected, query = s.recommend(history)
        assert len(set(selected)) == 5
        assert s.pairs == tradeoff_pairs(X)
        if query is None:
            break
        assert query not in {q for q, _ in history}
        history += ((query, 2),)
    if len(history) == 3:
        assert s.recommend(history)[1] is None


def test_trained_artifact_and_saved_sources_unchanged():
    saved = HERE/"artifact-v1"
    manifest = json.loads((saved/"manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["trained_rows"] == 1152 and manifest["train_states"] == 144
    for name, expected in manifest["artifact_sha256"].items():
        assert sha256((saved/name).read_bytes()).hexdigest() == expected
    fit = json.loads((saved/"fit-final.json").read_text())
    assert fit["steps"] == 400
    assert fit["losses"][-1]["standardized_train_mse"] < fit["losses"][0]["standardized_train_mse"]
    assert fit["numpy_torch_max_abs_difference"] < 1e-6


def test_dataset_splits_and_observable_input_schema():
    groups = [set(TRAIN_SEEDS), set(AGGREGATION_SEEDS), set(DEV_SEEDS), set(TINY_SEEDS)]
    for i, group in enumerate(groups):
        assert not group & {202610081000, 202610081001}
        for other in groups[i+1:]:
            assert not group & other
    assert all(202610082000 <= seed <= 202610082099 for group in groups for seed in group)
    assert not any("seed" in f or "true" in f or "label" in f for f in FEATURE_NAMES)
    assert tuple(inspect.signature(Session.features).parameters) == ("self", "history", "k")
    with np.load(HERE/"artifact-v1/teacher-initial.npz", allow_pickle=False) as saved:
        assert saved["features"].shape[1] == len(FEATURE_NAMES) == 41


def test_feature_no_one_step_checker_or_respondent_seed_dependency(monkeypatch):
    a = Session(X, Config(random_seed=123))
    b = Session(X, Config(random_seed=999))
    model = a.model()
    def forbidden(*args, **kwargs):
        raise AssertionError("feature extractor computed full KG")
    monkeypatch.setattr(FiniteModel, "one_step_values", forbidden)
    ids, features = a.features()
    other_ids, other = b.features()
    assert ids == other_ids
    np.testing.assert_array_equal(features, other)
    assert model.prior.shape[0] > 1


def test_skip_preserves_posterior_but_spends_slot():
    s = Session(X)
    model = s.model()
    np.testing.assert_array_equal(model.posterior(), model.posterior(((0, 2),)))
    ids, before = s.features()
    after_ids, after = s.features(((0, 2),))
    positions = {q: i for i, q in enumerate(ids)}
    np.testing.assert_array_equal(after[:, 4:], before[[positions[q] for q in after_ids], 4:])
    assert np.all(after[:, 0] == 2/3) and np.all(after[:, 3] == 1/3)


def test_history_order_no_double_conditioning():
    s = Session(X)
    history = ((0, 0), (1, 1))
    a = s.features(history)[1]
    b = s.features(tuple(reversed(history)))[1]
    np.testing.assert_array_equal(a, b)
    model = s.model()
    p = model.prior*model.likelihood[0, 0]*model.likelihood[1, 1]
    p /= p.sum()
    np.testing.assert_allclose(model.posterior(history), p, rtol=1e-13)


def test_candidate_permutation_features_predictions_and_selection():
    permutation = np.array([4,1,6,0,5,2,3])
    a, b = Session(X), Session(X[permutation])
    history = ((0, 0),)
    ids, features = a.features(history)
    bids, bfeatures = b.features(history)
    assert ids == bids
    np.testing.assert_allclose(features, bfeatures, rtol=0, atol=2e-15)
    np.testing.assert_allclose(a.predicted(history)[1], b.predicted(history)[1], rtol=0, atol=2e-14)
    chosen, q = a.recommend(history)
    bchosen, bq = b.recommend(history)
    assert tuple(permutation[list(bchosen)]) == chosen and q == bq


def test_fixed_seed_global_rng_and_replay():
    before = np.random.get_state()
    a = Session(X).recommend(((0, 0),))
    b = Session(X).recommend(((0, 0),))
    after = np.random.get_state()
    assert a == b
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_common_terminal_and_dominance():
    history = ((0, 0), (1, 2))
    selections = {Session(X, Config(mode=mode)).recommend(history)[0] for mode in MODES}
    assert len(selections) == 1
    selected = next(iter(selections))
    for excluded in set(range(len(X)))-set(selected):
        for included in selected:
            assert not (np.all(X[excluded] >= X[included]) and np.any(X[excluded] > X[included]))


def test_equal_budget_random_and_learned_checker():
    for history in ((), ((0, 0),)):
        a = Session(X, Config(mode="learned_lookahead"))
        b = Session(X, Config(mode="random_lookahead"))
        a.recommend(history)
        b.recommend(history)
        assert a.last_diagnostics["checked_questions"] == b.last_diagnostics["checked_questions"] == 4
        assert a.last_diagnostics["checker_depth"] == b.last_diagnostics["checker_depth"]
        assert a.last_diagnostics["omitted_action_gap_certificate"] is None


def test_last_slot_all_question_control():
    history = ((0, 0), (1, 2))
    result = None
    for mode in ("learned_lookahead", "random_lookahead", "teacher", "greedy"):
        s = Session(X, Config(mode=mode))
        current = s.recommend(history)
        assert result is None or current == result
        result = current
        assert s.last_diagnostics["pool_coverage"] == 1.


def test_guarded_kg_equals_greedy():
    for history in ((), ((0, 0),), ((0, 1), (1, 2))):
        a = Session(X, Config(mode="greedy"))
        b = Session(X, Config(mode="guarded_kg"))
        assert a.recommend(history) == b.recommend(history)
        assert abs(a.last_diagnostics["chosen_checker_value"]-b.last_diagnostics["chosen_checker_value"]) < 1e-14


def test_forced_root_against_independent_small_enumeration():
    u = np.array([[1,0,.5,.2,.8,.3], [0,1,.5,.8,.2,.3],
                  [.2,.8,1,.3,.5,.2], [.8,.2,0,.5,.3,1.]])
    left = np.array([[.8,.2,.7,.3], [.3,.8,.2,.9], [.4,.6,.9,.1]])
    likelihood = np.stack((.8*left, .8*(1-left), np.full_like(left, .2)), axis=1)
    model = FiniteModel(u, likelihood, np.full(4, .25), 5)
    def terminal(mass):
        return sum(sorted(mass @ u)[-5:])
    def recursive(mass, remaining, budget):
        stop = terminal(mass)
        if budget == 0:
            return stop
        return max([stop]+[sum(recursive(mass*likelihood[q, r],
                                               tuple(j for j in remaining if j != q), budget-1)
                                    for r in range(3)) for q in remaining])
    observed, _ = action_values(model, (), (0,1,2), depth=3, beam=None)
    expected = [sum(recursive(model.prior*likelihood[q, r], tuple(j for j in range(3) if j != q), 2)
                    for r in range(3)) for q in range(3)]
    np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-14)


@pytest.mark.parametrize("values", [[], [[np.nan]], [[1.1]], [[.1]*4]])
def test_invalid_values(values):
    with pytest.raises(ValueError):
        Session(values)


@pytest.mark.parametrize("history", [((0,0),(0,1)), ((-1,0),), ((0,3),),
                                      ((0,0),(1,0),(2,0),(3,0))])
def test_invalid_history(history):
    with pytest.raises(ValueError):
        Session(X).recommend(history)


@pytest.mark.parametrize("changes", [{"m":0}, {"mode":"hidden"}, {"shortlist":0},
                                        {"temperature":0}, {"skip_probability":1}])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        Config(**changes)


def test_small_duplicate_pool_and_unknown_acquisition_file(tmp_path):
    s = Session([[.5,.5]]*3)
    assert s.recommend() == ((0,1,2), None)
    file = tmp_path/"bad.npz"
    np.savez(file, format=np.array(1))
    with pytest.raises(ValueError):
        Student(file)


def test_negative_result_against_greedy_preserved():
    summary = json.loads((HERE/"artifact-v1/dev-summary.json").read_text())["modes"]
    assert summary["learned_lookahead"]["mean_signed_teacher_gap"] > summary["greedy"]["mean_signed_teacher_gap"]
    assert summary["guarded_kg"]["mean_global_one_step_loss"] == 0.
