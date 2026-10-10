import os
os.environ.setdefault("FSP_SECRET_KEY", "test-only-key-at-least-32-characters-2026")
os.environ["FSP_DEMO_MODE"] = "true"

import pytest
import uuid
from fastapi.testclient import TestClient
from sqlalchemy import event, text
from sqlmodel import SQLModel, Session, create_engine
from app.core.db import get_db
from app.core.security import hash_password, issue_access
from app.core import rate_limit
from app.main import app
from app.models import User, Candidate, Company, now

PASSWORD = "OnlyForTests2026!"
HASH = hash_password(PASSWORD)


@pytest.fixture
def environment(tmp_path):
    pg_url = os.environ.get("FSP_TEST_DATABASE_URL")
    admin = None
    schema = "fsp_test_" + uuid.uuid4().hex
    if pg_url:
        admin = create_engine(pg_url)
        with admin.begin() as conn: conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_engine(pg_url, connect_args={"options": f"-csearch_path={schema}"})
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False, "timeout": 20})
        @event.listens_for(engine, "connect")
        def pragma(conn, _):
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
    SQLModel.metadata.create_all(engine)
    def override():
        with Session(engine) as db: yield db
    app.dependency_overrides[get_db] = override
    rate_limit._entries.clear()
    client = TestClient(app)  # No lifespan: never touches the developer/demo database.
    yield client, engine
    client.close()
    app.dependency_overrides.clear()
    engine.dispose()
    if admin:
        with admin.begin() as conn: conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def actors(environment):
    client, engine = environment
    output = {}
    with Session(engine) as db:
        for name, role in (("candidate", "candidate"), ("other", "candidate"), ("employer", "employer"), ("employer2", "employer")):
            u = User(email=f"{name}@example.org", full_name=f"Private {name}", hashed_password=HASH, role=role, verified=True)
            db.add(u); db.flush()
            if role == "candidate":
                db.add(Candidate(user_id=u.id, headline="Python backend", about="Опыт создания API", city="Москва",
                    phone="+7 111 222 33 44", telegram="private_handle", published=True,
                    processing_consent_at=now(), publication_consent_at=now(), specialization="backend", grade="Junior",
                    salary_min=100000, salary_max=160000, skills={s: {"score": 100, "families": 2, "checked_at": now(), "source": "test"} for s in ["Python", "SQL", "API"]}))
            else:
                db.add(Company(user_id=u.id, name=f"Company {name}", description="Test company", contact=u.email))
            output[name] = {"id": u.id, "email": u.email, "headers": {"Authorization": f"Bearer {issue_access(u)}"}}
        db.commit()
    return output
