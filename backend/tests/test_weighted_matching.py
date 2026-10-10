"""Explicit weights must not alter eligibility, grades, ownership or contact grants."""
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from sqlmodel import Session

from app.models import Candidate, now
from app.schemas import SearchInput
from app.weighted_matching import prepare_weighted, select_weighted


def payload(weights=None, **extra):
    return {"specialization": "backend", "grade": "Junior", "ranking": {
        "mode": "weighted", "weights": weights or {},
    }, **extra}


def test_scoring_copy_matches_checked_prototype():
    root = Path(__file__).resolve().parents[2]
    source = root / "validation/direct_weights/prototype.py"
    if source.exists():
        assert (root / "backend/app/weighted_scoring.py").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("ranking", [
    {"mode": "weighted", "weights": {"SQL": 0}},
    {"mode": "weighted", "weights": {"SQL": -1}},
    {"mode": "weighted", "weights": {"SQL": 101}},
    {"mode": "weighted", "weights": {"SQL": True}},
    {"mode": "weighted", "weights": {"SQL": "20"}},
    {"mode": "weighted", "weights": {"SQL": float("nan")}},
    {"mode": "weighted", "weights": {"SQL": float("inf")}},
    {"mode": "preference", "weights": {"SQL": 20}},
    {"mode": "evidence", "weights": {"SQL": 20}},
    {"mode": "robust", "weights": {"SQL": 20}},
    {"mode": "weighted", "priorities": [["Python", "SQL"]]},
    {"mode": "weighted", "comparisons": [{"left": {"SQL": 50}, "right": {"SQL": 100}, "choice": "left"}]},
])
def test_input_rejects_invalid_or_mixed_modes(ranking):
    with pytest.raises(ValidationError):
        SearchInput(ranking=ranking)


def test_adapter_tie_rule_grade_immutability_and_large_pool():
    candidates = [Candidate(user_id=f"weight-{i:03}", grade="Middle", specialization="backend", published=True,
                            skills={s: {"score": 75, "families": 4, "checked_at": now()} for s in ("Python", "SQL", "API")})
                  for i in range(101)]
    candidates[-1].achievements = [{"verified": True, "source": "Тестовый реестр", "revoked": False}]
    before = [c.model_dump() for c in candidates]
    query = SearchInput(**payload({"Python": 2, "SQL": 1, "API": 1}))
    prepare_weighted(query)
    rows, report = select_weighted([(c, (0, [], [])) for c in candidates], query)
    assert report.considered == 101 and report.selected_count == 5
    assert report.weights == {"Python": .5, "SQL": .25, "API": .25}
    assert [c.user_id for c, _ in rows] == ["weight-100", "weight-000", "weight-001", "weight-002", "weight-003"]
    assert all(score == 75 for _, (score, _, _) in rows)
    assert [c.model_dump() for c in candidates] == before
    candidates[-1].achievements = [{"verified": True, "source": "Тестовый реестр", "revoked": True}]
    rows, _ = select_weighted([(c, (0, [], [])) for c in candidates], query)
    assert [c.user_id for c, _ in rows] == [f"weight-{i:03}" for i in range(5)]


def test_api_weights_change_order_save_restore_and_do_not_open_contacts(environment, actors):
    client, engine = environment
    with Session(engine) as db:
        for actor, scores in (("candidate", [100, 100, 50]), ("other", [100, 50, 100])):
            candidate = db.get(Candidate, actors[actor]["id"])
            candidate.skills = {s: {"score": score, "families": 4, "checked_at": now()} for s, score in zip(("Python", "SQL", "API"), scores)}
            db.add(candidate)
        db.commit()
    h = actors["employer"]["headers"]
    first = client.post("/api/search", headers=h, json=payload({"Python": 10, "SQL": 80, "API": 10}))
    assert first.status_code == 200, first.text
    assert first.json()["algorithm"] == "direct-weights-v1"
    assert first.json()["candidates"][0]["id"] == actors["candidate"]["id"]
    data = payload({"Python": 10, "SQL": 10, "API": 80})
    second = client.post("/api/search", headers=h, json=data)
    assert second.status_code == 200, second.text
    result = second.json()
    assert result["candidates"][0]["id"] == actors["other"]["id"]
    assert [c["match_score"] for c in result["candidates"]] == [95., 60.]
    assert result["preference"] is None and result["robust"] is None
    assert all(c["contacts"] is None and c["grade"] == "Junior" for c in result["candidates"])
    for secret in ("private_handle", "+7 111", "@example.org", "Private"):
        assert secret not in second.text
    saved = client.post("/api/needs", headers=h, json=data)
    assert saved.status_code == 201, saved.text
    need_id = saved.json()["id"]
    assert saved.json()["ranking"]["weights"] == data["ranking"]["weights"]
    restored = client.get(f"/api/needs/{need_id}", headers=h)
    assert restored.status_code == 200, restored.text
    assert restored.json()["current"]["weighted"] == result["weighted"]
    assert [c["id"] for c in restored.json()["previous"]] == [c["id"] for c in result["candidates"]]
    assert client.get(f"/api/needs/{need_id}", headers=actors["employer2"]["headers"]).status_code == 404
    assert client.post("/api/search", headers=actors["candidate"]["headers"], json=data).status_code == 403
    offer = {"candidate_id": actors["candidate"]["id"], "title": "Роль Python backend",
             "description": "Приглашаем обсудить роль в тестовой команде.", "salary_min": 110000,
             "salary_max": 160000, "currency": "RUB", "contact": "contact@company.invalid"}
    invitation = client.post("/api/invitations", headers=h, json=offer)
    assert invitation.status_code == 201, invitation.text
    action = client.post(f"/api/invitations/{invitation.json()['id']}/action", headers=actors["candidate"]["headers"], json={"action": "accept"})
    assert action.status_code == 200, action.text
    own = client.post("/api/search", headers=h, json=data).json()
    assert next(c for c in own["candidates"] if c["id"] == actors["candidate"]["id"])["contacts"]["email"] == actors["candidate"]["email"]
    other = client.post("/api/search", headers=actors["employer2"]["headers"], json=data).json()
    assert all(c["contacts"] is None for c in other["candidates"])
    with Session(engine) as db:
        hidden = db.get(Candidate, actors["other"]["id"])
        hidden.published = False
        db.add(hidden); db.commit()
    restored = client.get(f"/api/needs/{need_id}", headers=h).json()
    assert all(c["id"] != actors["other"]["id"] for c in restored["previous"] + restored["current"]["candidates"])


def test_zero_weight_does_not_remove_required_skill_or_salary_filter(environment, actors):
    client, engine = environment
    with Session(engine) as db:
        other = db.get(Candidate, actors["other"]["id"])
        other.skills = {s: v for s, v in other.skills.items() if s != "SQL"}
        db.add(other); db.commit()
    h = actors["employer"]["headers"]
    data = payload({"Python": 100, "SQL": 0, "API": 0})
    response = client.post("/api/search", headers=h, json=data)
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    assert response.json()["candidates"][0]["id"] == actors["candidate"]["id"]
    data.update(salary_min=400000, salary_max=500000)
    response = client.post("/api/search", headers=h, json=data)
    assert response.status_code == 200 and response.json()["total"] == 0
    assert response.json()["weighted"]["selected_count"] == 0


def test_default_equal_weights_unknown_keys_and_mode_switch(environment, actors):
    client, _ = environment
    h = actors["employer"]["headers"]
    data = client.post("/api/search", headers=h, json=payload()).json()
    assert data["normalized"]["ranking"]["weights"] == dict.fromkeys(("Python", "SQL", "API"), 1.)
    np.testing.assert_allclose(list(data["weighted"]["weights"].values()), [1/3] * 3)
    for weights in ({"SQL": 20}, {"Python": 10, "SQL": 10, "Rust": 80}):
        assert client.post("/api/search", headers=h, json=payload(weights)).status_code == 422
    assert client.post("/api/search", headers=h, json={"ranking": {"mode": "weighted"}}).status_code == 422
    normalized = data["normalized"]
    normalized["ranking"] = {"mode": "preference", "comparisons": []}
    result = client.post("/api/search", headers=h, json=normalized)
    assert result.status_code == 200 and result.json()["weighted"] is None
    assert result.json()["preference"]["answered"] == 0
