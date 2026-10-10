from copy import deepcopy

import pytest
from sqlmodel import Session

from app.matching import evaluate
from app.mixed_matching import select_mixed
from app.models import Candidate, User, now
from app.preference_matching import prepare_preference
from app.schemas import SearchInput


def profiles():
    rows = [[100,100,100]]*4 + [[100,50,100], [50,100,100], [100,100,50], [75,75,75]]
    return [Candidate(user_id=f"m6-{i}", grade="Middle", specialization="backend", published=True,
        processing_consent_at=now(), publication_consent_at=now(), phone="private-m6-phone",
        skills={s: {"score": v, "families": 2, "checked_at": now()}
                for s,v in zip(("Python", "SQL", "API"), row)}) for i,row in enumerate(rows)]


def run(candidates, answers=(), skills=()):
    q = SearchInput(specialization="backend", skills=list(skills),
                    ranking={"mode": "mixed", "comparisons": list(answers)})
    prepare_preference(q)
    return select_mixed([(c, evaluate(c, q)) for c in candidates], q)


def test_mixed_three_slots_skip_undo_and_no_mutation():
    candidates = profiles()
    before = [c.model_dump() for c in candidates]
    first, report = run(candidates)
    ids = [c.user_id for c,_ in first]
    history = []
    for choice in ("skip", "left", "right"):
        assert report.question is not None
        history.append({"left": report.question.left, "right": report.question.right, "choice": choice})
        rows, report = run(candidates, history)
        if choice == "skip":
            assert [c.user_id for c,_ in rows] == ids
    assert report.remaining == 0 and report.question is None
    assert report.engine == "mixed-kg-v1" and report.weights == {}
    restored, _ = run(candidates)
    assert [c.user_id for c,_ in restored] == ids
    assert [c.model_dump() for c in candidates] == before


def test_mixed_skill_order_and_candidate_order_do_not_change_result():
    a, ra = run(profiles(), skills=("Python", "SQL", "API"))
    b, rb = run(list(reversed(profiles())), skills=("API", "Python", "SQL"))
    assert [c.user_id for c,_ in a] == [c.user_id for c,_ in b]
    assert ra.question == rb.question


def test_fsp_only_breaks_identical_skill_tie():
    candidates = profiles()[:4]
    candidates += [deepcopy(candidates[0]) for _ in range(2)]
    candidates[-2].user_id = "m6-4"
    candidates[-1].user_id = "m6-5"
    candidates[-1].fsp_id = "NOT-A-VERIFICATION"
    rows, _ = run(candidates)
    assert "m6-5" not in [c.user_id for c,_ in rows]
    candidates[-1].achievements = [{"verified": True, "source": "fictional-test-source"}]
    rows, _ = run(candidates)
    assert "m6-5" in [c.user_id for c,_ in rows]


def test_api_mixed_save_restore_privacy_and_changed_pool(environment, actors):
    client, engine = environment
    with Session(engine) as db:
        for candidate in profiles():
            db.add(User(id=candidate.user_id, email=f"{candidate.user_id}@example.org",
                        full_name="Fictional M6", hashed_password="disabled", role="candidate", verified=True))
            db.flush()
            db.add(candidate)
        db.commit()
    headers = actors["employer"]["headers"]
    payload = {"specialization": "backend", "grade": "Middle", "ranking": {"mode": "mixed"}}
    first = client.post("/api/search", headers=headers, json=payload)
    assert first.status_code == 200, first.text
    data = first.json()
    assert data["algorithm"] == "mixed-kg-v1"
    assert "private-m6-phone" not in first.text and "@example.org" not in first.text
    assert all(c["contacts"] is None for c in data["candidates"])
    q = data["preference"]["question"]
    payload["ranking"]["comparisons"] = [{"left": q["left"], "right": q["right"], "choice": "left"}]
    saved = client.post("/api/needs", headers=headers, json=payload)
    assert saved.status_code == 201, saved.text
    path = f"/api/needs/{saved.json()['id']}"
    restored = client.get(path, headers=headers)
    assert restored.status_code == 200
    assert restored.json()["current"]["algorithm"] == "mixed-kg-v1"
    assert restored.json()["current"]["preference"]["answered"] == 1
    assert client.get(path, headers=actors["employer2"]["headers"]).status_code == 404
    with Session(engine) as db:
        candidate = db.get(Candidate, "m6-4")
        candidate.published = False
        db.add(candidate); db.commit()
    updated = client.get(path, headers=headers)
    assert updated.status_code == 200, updated.text
    assert all(c["id"] != "m6-4" for c in updated.json()["current"]["candidates"])
    assert client.post("/api/search", headers=actors["candidate"]["headers"], json=payload).status_code == 403


@pytest.mark.parametrize("payload", [
    {"ranking": {"mode": "mixed"}},
    {"skills": ["Rust"], "ranking": {"mode": "mixed"}},
    {"specialization": "backend", "ranking": {"mode": "mixed", "comparisons": [
        {"left": {"Python": 100., "SQL": 100., "API": 100.},
         "right": {"Python": 50., "SQL": 50., "API": 50.}, "choice": "left"}]}}
])
def test_bad_mixed_request_is_rejected(environment, actors, payload):
    client, _ = environment
    assert client.post("/api/search", headers=actors["employer"]["headers"], json=payload).status_code == 422
