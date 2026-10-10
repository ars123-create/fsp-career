"""Independent adapter checks; public API tests are enabled at integration."""
from pathlib import Path
from hashlib import sha256
import json

import numpy as np
import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.models import Candidate, User, now
from app.matching import evaluate
from app.robust_matching import prepare_robust, select_robust
from app.schemas import SearchInput
from app.robust_engine import WeightRegion, worst_case


def test_engine_matches_frozen_research_source():
    root = Path(__file__).resolve().parents[2]
    core = (root / "backend/app/robust_engine.py").read_bytes()
    manifest = json.loads((root / "backend/algorithm_manifest.json").read_text())
    assert sha256(core).hexdigest() == manifest["engine_sha256"]
    research = root / "research/robust_topk/core.py"
    if research.exists():
        assert core == research.read_bytes()


def candidate(i, scores, **kw):
    return Candidate(user_id=str(i), grade="Junior", specialization="backend", published=True,
        skills={skill: {"score": score, "families": 2, "checked_at": now()} for skill,score in zip(("Python", "SQL", "API"), scores)}, **kw)


def test_adapter_certificate_and_guard_against_observed_scores():
    profiles = [candidate(i, row) for i, row in enumerate(np.random.default_rng(478).uniform(50,100,(18,3)))]
    q = SearchInput(specialization="backend", ranking={"mode": "robust"})
    region = prepare_robust(q)
    ranked = [(c, evaluate(c, q)) for c in profiles]
    selected, report = select_robust(ranked, q, region)
    assert len(selected) == report.selected_count == 5
    assert report.considered == 18 and report.status == "optimal"
    assert report.worst_loss_points <= report.baseline_worst_loss_points + 1e-5
    assert report.nominal_loss_points <= 1.00001
    assert all(c.grade == "Junior" and c.published for c in profiles)
    # Check the emitted witness using independent hand-computed feature vectors.
    matrix = np.array([[.8*c.skills[s]["score"]/100+.1+.05 for s in q.skills] for c in profiles])
    picked = tuple(int(c.user_id) for c,_ in selected)
    expected = worst_case(matrix, matrix, region.vertices(), picked)
    assert report.worst_loss_points == pytest.approx(expected.regret*100/5, abs=1e-4)


def test_invalid_priority_and_ambiguous_skill_space():
    with pytest.raises(HTTPException):
        prepare_robust(SearchInput(ranking={"mode":"robust"}))
    with pytest.raises(HTTPException):
        prepare_robust(SearchInput(skills=["Python"], ranking={"mode":"robust", "priorities":[["Python","SQL"]]}))
    with pytest.raises(HTTPException):
        prepare_robust(SearchInput(skills=["Rust"], ranking={"mode":"robust"}))


def test_guard_anchor_survives_preference_change():
    profiles = [candidate(i, row) for i, row in enumerate(np.random.default_rng(479).uniform(50,100,(18,3)))]
    bounds = []
    for pairs in ([], [["Python", "SQL"]], [["Python", "SQL"], ["SQL", "API"]]):
        q = SearchInput(specialization="backend", ranking={"mode":"robust", "priorities":pairs})
        region = prepare_robust(q)
        selected, report = select_robust([(c,evaluate(c,q)) for c in profiles],q,region)
        bounds.append(report.worst_loss_points)
        assert report.nominal_loss_points <= 1.00001
    assert bounds[2] <= bounds[1] + 1e-5 <= bounds[0] + 2e-5


def test_api_filters_privacy_and_saved_priorities(environment, actors):
    client, engine = environment
    h = actors["employer"]["headers"]
    with Session(engine) as db:
        person = db.get(Candidate, actors["other"]["id"])
        person.salary_min = 500000
        person.salary_max = 600000
        db.add(person)
        db.commit()
    payload = {"specialization":"backend", "salary_min":100000,"salary_max":180000,
               "ranking":{"mode":"robust", "priorities":[["Python","SQL"]]}}
    response = client.post("/api/search",headers=h,json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["algorithm"] == "guarded-minimax-v2"
    assert result["total"] == 1
    assert result["candidates"][0]["contacts"] is None
    assert result["robust"]["challenger_ids"] == [actors["candidate"]["id"]]
    assert all(secret not in response.text for secret in ("private_handle", actors["candidate"]["email"], "Private candidate"))
    assert client.post("/api/search",headers=actors["candidate"]["headers"],json=payload).status_code == 403
    saved = client.post("/api/needs",headers=h,json=payload)
    assert saved.status_code == 201, saved.text
    need = saved.json()
    assert need["ranking"] == {**payload["ranking"], "comparisons": []}
    restored = client.get(f"/api/needs/{need['id']}", headers=h).json()
    assert restored["current"]["normalized"]["ranking"] == {**payload["ranking"], "comparisons": []}
    assert client.get(f"/api/needs/{need['id']}",headers=actors["employer2"]["headers"]).status_code == 404
    with Session(engine) as db:
        person = db.get(Candidate, actors["candidate"]["id"])
        assert person.grade == "Junior"
        person.published=False
        db.add(person)
        db.commit()
    restored = client.get(f"/api/needs/{need['id']}", headers=h).json()
    assert restored["previous"] == restored["current"]["candidates"] == []
    assert restored["current"]["robust"] is None


def test_api_rejects_bad_priority_and_keeps_default_mode(environment, actors):
    client,_ = environment
    h=actors["employer"]["headers"]
    assert client.post("/api/search",headers=h,json={}).json()["algorithm"] == "evidence-v1"
    for payload in ({"ranking":{"mode":"robust"}},
                    {"skills":["Python"],"ranking":{"mode":"robust","priorities":[["SQL","API"]]}}):
        assert client.post("/api/search",headers=h,json=payload).status_code == 422
