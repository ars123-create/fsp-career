"""Scenario checks for the adapter, then API ownership and contact boundaries."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlmodel import Session

from app.models import Candidate, User, now
from app.matching import evaluate
from app.preference_matching import pair_key, prepare_preference, select_preference
from app.preference_engine import simplex_particles
from app.schemas import SearchInput


def profiles():
    # Four clear leaders and competing skill patterns at the shortlist boundary.
    scores = [[100,100,100]]*4 + [[100,50,100], [50,100,100], [100,100,50], [75,75,75]]
    return [Candidate(user_id=f"p{i}", grade="Middle", specialization="backend", published=True,
        salary_min=100000, salary_max=180000,
        skills={s:{"score":score,"families":2,"checked_at":now()}
                for s,score in zip(("Python","SQL","API"), row)}) for i,row in enumerate(scores)]


def run(candidates, comparisons=(), **kw):
    q = SearchInput(specialization="backend", ranking={"mode":"preference", "comparisons":list(comparisons)}, **kw)
    prepare_preference(q)
    rows, report = select_preference([(c, evaluate(c,q)) for c in candidates],q)
    return q, rows, report


def answer(question, choice="left"):
    return {"left":question.left,"right":question.right,"choice":choice}


def test_engine_exact_copy_of_frozen_research():
    root=Path(__file__).resolve().parents[2]
    source=(root/"backend/app/preference_engine.py").read_bytes()
    manifest=json.loads((root/"backend/preference_manifest.json").read_text())
    assert sha256(source).hexdigest()==manifest["engine_sha256"]
    research=root/"research/preference_topk/core.py"
    if research.exists():
        assert source==research.read_bytes()


@pytest.mark.parametrize("n,seed", [(18, 934021), (18, 934022), (100, 934021), (100, 934022)])
def test_accelerated_adapter_matches_reference_reports(monkeypatch, n, seed):
    import app.preference_matching as adapter
    from app.preference_engine import Belief, questions
    rng = np.random.default_rng(seed)
    candidates = [Candidate(user_id=f"p{i:03}", grade="Middle", specialization="backend", published=True,
        skills={skill: {"score": float(value), "families": 4, "checked_at": now()}
                for skill, value in zip(("Python", "SQL", "API"), row)})
        for i, row in enumerate(rng.uniform(50, 100, (n, 3)))]
    def reference_picker(values, weights, probabilities, k, *, asked, min_gain):
        options = questions(Belief(weights @ values.T, probabilities), k, asked=asked)
        best = max(options, key=lambda q: q.expected_gain) if options else None
        return best if best and best.expected_gain > min_gain else None
    history = []
    for choice in ("left", "right", "skip"):
        _, actual_rows, actual = run(candidates, history)
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "best_linear_question", reference_picker)
            _, expected_rows, expected = run(candidates, history)
        assert [c.user_id for c, _ in actual_rows] == [c.user_id for c, _ in expected_rows]
        assert [score for _, score in actual_rows] == [score for _, score in expected_rows]
        left, right = actual.model_dump(), expected.model_dump()
        if actual.question is not None:
            assert left["question"].pop("expected_gain_points") == pytest.approx(
                right["question"].pop("expected_gain_points"), abs=1e-10)
        assert left == right
        if actual.question is None:
            break
        history.append(answer(actual.question, choice))


def test_adapter_matches_manual_bayes_update_and_leaves_grade_unchanged():
    candidates=profiles()
    q, initial, report=run(candidates)
    assert report.status=="question" and report.question is not None
    comparison=answer(report.question)
    q, rows, after=run(candidates,[comparison])
    weights=simplex_particles(3)
    difference=np.array([comparison["left"][s]-comparison["right"][s] for s in q.skills])/100
    likelihood=.01+.98/(1+np.exp(-(weights@difference)/.04))
    p=likelihood/likelihood.sum()
    means=p@weights
    matrix=np.array([[c.skills[s]["score"] for s in q.skills] for c in candidates])
    scores=matrix@means
    expected=np.argsort(-scores,kind="stable")[:5]
    assert [c.user_id for c,_ in rows]==[candidates[i].user_id for i in expected]
    np.testing.assert_allclose(list(after.weights.values()),means,atol=1e-12)
    assert [score for _,(score,_,_) in rows]==[round(float(scores[i]),1) for i in expected]
    assert all(c.grade=="Middle" for c in candidates)
    assert after.answered==1 and after.skipped==0 and after.remaining==2
    assert report.considered==8 and report.selected_count==5


def test_skip_undo_repeated_vectors_and_budget():
    candidates=profiles()
    _, initial, report=run(candidates)
    assert report.changed_count == 0
    history=[]
    seen=set()
    for _ in range(3):
        if report.question is None:
            break
        key=pair_key(report.question.left,report.question.right)
        differences = [report.question.left[s]-report.question.right[s] for s in report.weights]
        assert min(differences) < 0 < max(differences)
        assert key not in seen
        seen.add(key)
        history.append(answer(report.question,"skip"))
        _,rows,report=run(candidates,history)
        assert [c.user_id for c,_ in rows]==[c.user_id for c,_ in initial]
        assert report.changed_count == 0
        np.testing.assert_allclose(list(report.weights.values()),[1/3]*3,atol=1e-12)
    assert report.answered==0 and report.skipped==len(history)
    assert report.question is None or len(history)<3
    _,again,reset=run(candidates)
    assert reset.answered==reset.skipped==0 and reset.remaining==3
    assert [c.user_id for c,_ in again]==[c.user_id for c,_ in initial]


def test_equal_prior_ties_and_no_dominated_questions():
    candidates=profiles()
    _,rows,report=run(candidates)
    assert [c.user_id for c,_ in rows]==["p0","p1","p2","p3","p4"]
    # No meaningful trade-off if everyone differs in a single skill only.
    for i,c in enumerate(candidates):
        c.skills["SQL"]["score"]=100
        c.skills["API"]["score"]=100
        c.skills["Python"]["score"]=50+i*5
    _,_,report=run(candidates)
    assert report.question is None and report.status=="not_needed"


def test_fsp_ties_preserve_question_model_skip_and_primary_scores():
    from copy import deepcopy
    plain=profiles()
    with_fsp=deepcopy(plain)
    with_fsp[6].fsp_id="DEMO-CLAIM"
    with_fsp[6].achievements=[{"verified":True,"revoked":False,"source":"Тестовый реестр","is_demo":True}]
    original=[c.model_dump() for c in with_fsp]
    _,before,root=run(plain)
    _,initial,with_report=run(with_fsp)
    assert [c.user_id for c,_ in before]==["p0","p1","p2","p3","p4"]
    assert [c.user_id for c,_ in initial]==["p0","p1","p2","p3","p6"]
    assert with_report.question==root.question and with_report.weights==root.weights
    for choice in ("left","right","skip"):
        history=[answer(root.question,choice)]
        _,reference_rows,reference=run(plain,history)
        _,actual_rows,actual=run(with_fsp,history)
        assert actual.weights==reference.weights
        assert actual.question==reference.question
        assert actual.status==reference.status
        assert [points for _,(points,_,_) in actual_rows]==[points for _,(points,_,_) in reference_rows]
        if choice=="skip":
            assert [c.user_id for c,_ in actual_rows]==[c.user_id for c,_ in initial]
            assert actual.changed_count==0
    _,reset,report=run(with_fsp)
    assert [c.user_id for c,_ in reset]==[c.user_id for c,_ in initial]
    assert report.changed_count==0
    assert [c.model_dump() for c in with_fsp]==original


def test_empty_single_skill_small_pool_and_caps():
    _,rows,report=run([])
    assert not rows and report.status=="empty" and report.question is None
    _,rows,report=run(profiles()[:3])
    assert len(rows)==3 and report.question is None
    _,_,report=run(profiles(),skills=["Python"])
    assert report.question is None and report.weights=={"Python":1.}
    many=profiles()*13
    with pytest.raises(HTTPException) as e:
        run(many)
    assert e.value.status_code==422


def test_invalid_scales_duplicate_pairs_and_changed_skills():
    _,_,report=run(profiles())
    a=answer(report.question)
    reverse={"left":a["right"],"right":a["left"],"choice":"right"}
    for history in ([a,reverse],[{"left":{"SQL":100},"right":{"SQL":50},"choice":"left"}]):
        with pytest.raises(HTTPException):
            run(profiles(),history)
    for value in (-1,101,float("nan"),True):
        broken={**a,"left":{**a["left"],"Python":value}}
        with pytest.raises(ValidationError):
            run(profiles(),[broken])
    with pytest.raises(ValidationError):
        run(profiles(),[a]*4)
    with pytest.raises(HTTPException):
        run(profiles(),skills=["Rust"])


def add_profiles(engine, actors):
    with Session(engine) as db:
        for item in profiles():
            user=User(id=item.user_id,email=f"{item.user_id}@fiction.invalid",full_name=f"Private {item.user_id}",
                      role="candidate",verified=True,hashed_password="not-a-login-hash")
            db.add(user)
            db.flush()
            item.phone="+7 999 333 44 55"
            item.telegram="private_comparison_contact"
            item.processing_consent_at=item.publication_consent_at=now()
            db.add(item)
        db.commit()


def test_api_saved_answers_ownership_privacy_and_updated_candidate_pool(environment, actors):
    client,engine=environment
    add_profiles(engine, actors)
    headers=actors["employer"]["headers"]
    payload={"specialization":"backend","grade":"Middle","salary_min":100000,"salary_max":180000,
             "ranking":{"mode":"preference"}}
    response=client.post("/api/search",headers=headers,json=payload)
    assert response.status_code==200, response.text
    data=response.json()
    assert data["algorithm"]=="decision-evoi-v3" and data["total"]==8 and len(data["candidates"])==5
    assert data["robust"] is None and data["preference"]["question"]
    question=data["preference"]["question"]
    assert set(question)=={"left","right","expected_gain_points"}
    assert set(question["left"])=={"Python","SQL","API"}
    assert all(c["contacts"] is None for c in data["candidates"])
    for secret in ("private_comparison_contact","Private p","fiction.invalid","+7 999"):
        assert secret not in response.text
    payload=data["normalized"]
    payload["ranking"]["comparisons"].append({"left":question["left"],"right":question["right"],"choice":"left"})
    response=client.post("/api/search",headers=headers,json=payload)
    assert response.status_code==200
    after=response.json()
    assert after["preference"]["answered"]==1
    saved=client.post("/api/needs",headers=headers,json=payload)
    assert saved.status_code==201, saved.text
    need=saved.json()
    assert need["ranking"]["comparisons"]==payload["ranking"]["comparisons"]
    restored=client.get(f"/api/needs/{need['id']}",headers=headers).json()
    assert restored["current"]["preference"]["weights"]==after["preference"]["weights"]
    assert client.get(f"/api/needs/{need['id']}",headers=actors["employer2"]["headers"]).status_code==404
    assert client.post("/api/search",headers=actors["candidate"]["headers"],json=payload).status_code==403
    hidden=after["candidates"][0]["id"]
    with Session(engine) as db:
        candidate=db.get(Candidate,hidden)
        candidate.published=False
        db.add(candidate)
        db.commit()
    restored=client.get(f"/api/needs/{need['id']}",headers=headers).json()
    assert all(c["id"]!=hidden for c in restored["previous"]+restored["current"]["candidates"])
    assert restored["current"]["total"]==7
    # Answers contain only hypothetical numbers, not references to withdrawn people.
    assert hidden not in json.dumps(restored["need"]["ranking"])


def test_api_contact_consent_remains_specific_to_employer(environment, actors):
    client,_=environment
    h=actors["employer"]["headers"]
    data={"specialization":"backend","grade":"Junior","ranking":{"mode":"preference"}}
    candidate_id=actors["candidate"]["id"]
    result=client.post("/api/search",headers=h,json=data).json()
    assert all(c["contacts"] is None for c in result["candidates"])
    offer={"candidate_id":candidate_id,"title":"Обсудим работу Python backend",
           "description":"Приглашаем обсудить роль в тестовой команде.","salary_min":110000,
           "salary_max":160000,"currency":"RUB","contact":"contact@company.invalid"}
    created=client.post("/api/invitations",headers=h,json=offer)
    assert created.status_code==201, created.text
    accepted=client.post(f"/api/invitations/{created.json()['id']}/action",headers=actors["candidate"]["headers"],json={"action":"accept"})
    assert accepted.status_code==200, accepted.text
    allowed=client.post("/api/search",headers=h,json=data).json()
    own=next(c for c in allowed["candidates"] if c["id"]==candidate_id)
    assert own["contacts"]["email"]==actors["candidate"]["email"]
    other=client.post("/api/search",headers=actors["employer2"]["headers"],json=data).json()
    assert all(c["contacts"] is None for c in other["candidates"])


def test_api_fsp_tie_and_revocation_respect_saved_results_and_contacts(environment, actors):
    client,engine=environment
    with Session(engine) as db:
        for i in range(6):
            identifier=f"tie-{i}"
            db.add(User(id=identifier,email=f"{identifier}@fiction.invalid",full_name=f"Private {identifier}",
                        role="candidate",verified=True,is_demo=True,hashed_password="not-a-login-hash"))
            db.flush()
            db.add(Candidate(user_id=identifier,specialization="backend",grade="Middle",published=True,
                processing_consent_at=now(),publication_consent_at=now(),phone="private-tie-contact",
                skills={s:{"score":75,"families":4,"checked_at":now()} for s in ("Python","SQL","API")},
                fsp_id="DEMO-CLAIMED" if i==5 else "",
                achievements=[{"title":"Тестовое достижение · демо","verified":True,"revoked":False,
                               "source":"Тестовый реестр","is_demo":True}] if i==5 else []))
        db.commit()
    payload={"specialization":"backend","grade":"Middle","ranking":{"mode":"preference"}}
    h=actors["employer"]["headers"]
    for employer in ("employer","employer2"):
        response=client.post("/api/search",headers=actors[employer]["headers"],json=payload)
        assert response.status_code==200,response.text
        data=response.json()
        assert data["total"]==6
        assert [c["id"] for c in data["candidates"]]==["tie-5","tie-0","tie-1","tie-2","tie-3"]
        assert all(c["match_score"]==75 and c["contacts"] is None for c in data["candidates"])
        assert "private-tie-contact" not in response.text and "fiction.invalid" not in response.text
        assert any("При равной оценке" in notice for notice in data["notices"])
    saved=client.post("/api/needs",headers=h,json=payload)
    assert saved.status_code==201,saved.text
    with Session(engine) as db:
        candidate=db.get(Candidate,"tie-5")
        candidate.achievements=[{**candidate.achievements[0],"revoked":True}]
        db.add(candidate);db.commit()
    restored=client.get(f"/api/needs/{saved.json()['id']}",headers=h)
    assert restored.status_code==200,restored.text
    data=restored.json()
    assert [c["id"] for c in data["current"]["candidates"]]==[f"tie-{i}" for i in range(5)]
    assert data["current"]["total"]==6  # No FSP is still eligible for work.
    assert any(c["id"]=="tie-5" for c in data["previous"])
    assert all(c["grade"]=="Middle" and c["contacts"] is None for c in data["previous"]+data["current"]["candidates"])


def test_api_bad_preferences_and_incompatible_salary(environment, actors):
    client,_=environment
    h=actors["employer"]["headers"]
    for payload in ({"ranking":{"mode":"preference"}},
                    {"skills":["Rust"],"ranking":{"mode":"preference"}},
                    {"specialization":"backend","ranking":{"mode":"preference","priorities":[["Python","SQL"]]}}):
        assert client.post("/api/search",headers=h,json=payload).status_code==422
    response=client.post("/api/search",headers=h,json={"specialization":"backend","salary_min":400000,
                       "salary_max":500000,"ranking":{"mode":"preference"}})
    assert response.status_code==200
    assert response.json()["total"]==0 and response.json()["preference"]["status"]=="empty"
