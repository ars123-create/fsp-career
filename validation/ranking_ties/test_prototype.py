from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from validation.ranking_ties.prototype import evidence_tied_topk, has_confirmed_achievement


def profile(identifier, achievements=(), **kwargs):
    return SimpleNamespace(user_id=identifier, achievements=list(achievements), **kwargs)


def evidence(**kwargs):
    return {"verified": True, "revoked": False, "source": "Тестовый реестр", "is_demo": True, **kwargs}


def test_confirmed_achievement_at_top5_boundary():
    candidates=[profile(f"p{i}", [evidence()] if i==5 else []) for i in range(6)]
    assert evidence_tied_topk([.75]*6, candidates, 5)==(5,0,1,2,3)


def test_skill_score_always_precedes_achievement_at_existing_precision():
    candidates=[profile("a", [evidence()]), profile("z")]
    assert evidence_tied_topk([.75,.750000000001], candidates, 2)==(1,0)


@pytest.mark.parametrize("record", [
    {"verified":False,"source":"Тестовый реестр"},
    {"verified":True,"revoked":True,"source":"Тестовый реестр"},
    {"verified":True}, {"verified":True,"source":"  "},
    {"verified":"true","source":"Тестовый реестр"},
    {"verified":1,"source":"Тестовый реестр"},
    {"verified":True,"source":None},
])
def test_unconfirmed_or_malformed_evidence_cannot_promote(record):
    candidates=[profile("a"),profile("z",[record],fsp_id="FSP-CLAIMED")]
    assert not has_confirmed_achievement(candidates[1])
    assert evidence_tied_topk([.75,.75],candidates,1)==(0,)


def test_presence_not_count_and_no_private_or_id_text_preferences():
    candidates=[profile("a",[evidence()],fsp_id="",full_name="Я",phone=""),
                profile("z",[evidence()]*10,fsp_id="000001",full_name="А",phone="secret")]
    assert evidence_tied_topk([.75,.75],candidates,2)==(0,1)


def test_claimed_fsp_id_alone_cannot_promote():
    candidates=[profile("a",fsp_id=""),profile("z",fsp_id="FSP-CLAIMED")]
    assert evidence_tied_topk([.75,.75],candidates,1)==(0,)


def test_input_permutation_and_empty_result():
    candidates=[profile("a"),profile("b"),profile("z",[evidence()])]
    for order in permutations(candidates):
        indices=evidence_tied_topk([.75]*3,order,3)
        assert [order[i].user_id for i in indices]==["z","a","b"]
    assert evidence_tied_topk([],[],0)==()


def test_primary_topk_sum_and_input_immutability():
    for seed in range(934101,934151):
        rng=np.random.default_rng(seed)
        scores=rng.choice([.5,.75,1.],size=20)
        candidates=[profile(f"p{i:02}",[evidence()] if rng.integers(2) else []) for i in range(20)]
        originals=deepcopy([vars(c) for c in candidates]);old_scores=scores.copy()
        for k in range(1,21):
            indices=evidence_tied_topk(scores,candidates,k)
            assert scores[list(indices)].sum()==np.sort(scores)[-k:].sum()
        assert [vars(c) for c in candidates]==originals
        np.testing.assert_array_equal(scores,old_scores)
