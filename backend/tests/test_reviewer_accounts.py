"""Reviewer passwords are limited to fictional accounts on demo stands."""
from sqlmodel import Session, select
from app.seed import seed_demo, configure_demo_logins
from app.models import User, Invitation
from app.core.config import settings
from app.core.security import hash_password, verify_password


def test_reviewer_logins_and_repeat_seed_preserve_state(environment):
    client, engine = environment
    with Session(engine) as db:
        seed_demo(db)
        before_users = {u.id for u in db.exec(select(User)).all()}
        before_invites = {i.id for i in db.exec(select(Invitation)).all()}
        seed_demo(db)
        assert before_users == {u.id for u in db.exec(select(User)).all()}
        assert before_invites == {i.id for i in db.exec(select(Invitation)).all()}
    for role in ("candidate", "employer"):
        result = client.post("/api/auth/login", data={"username": f"{role}@demo.fsp.test", "password": "12345678"})
        assert result.status_code == 200
        assert result.json()["user"]["role"] == role
        auth = {"Authorization": "Bearer " + result.json()["access_token"]}
        assert client.get("/api/auth/me", headers=auth).status_code == 200
    assert client.post("/api/auth/login", data={"username": "employer@demo.fsp.test", "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/login", data={"username": "employer2@demo.fsp.test", "password": "12345678"}).status_code == 401


def test_demo_disabled_blocks_password_and_existing_token(environment, monkeypatch):
    client, engine = environment
    with Session(engine) as db:
        seed_demo(db)
    login = client.post("/api/auth/login", data={"username": "candidate@demo.fsp.test", "password": "12345678"})
    auth = {"Authorization": "Bearer " + login.json()["access_token"]}
    monkeypatch.setattr(settings, "demo_mode", False)
    assert client.post("/api/auth/login", data={"username": "candidate@demo.fsp.test", "password": "12345678"}).status_code == 401
    assert client.get("/api/auth/me", headers=auth).status_code == 401
    assert client.post("/api/auth/demo/candidate").status_code == 404


def test_no_seed_outside_demo(environment, monkeypatch):
    _, engine = environment
    monkeypatch.setattr(settings, "demo_mode", False)
    with Session(engine) as db:
        seed_demo(db)
        assert db.exec(select(User)).all() == []


def test_configuration_never_rewrites_non_demo_user(environment):
    _, engine = environment
    with Session(engine) as db:
        user = User(email="candidate@demo.fsp.test", full_name="Non demo", role="candidate", verified=True,
                    hashed_password=hash_password("Unchanged2026!"), is_demo=False)
        db.add(user); db.commit()
        configure_demo_logins(db)
        db.refresh(user)
        assert verify_password("Unchanged2026!", user.hashed_password)
