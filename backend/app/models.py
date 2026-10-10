"""SQLModel persistence; public API schemas never inherit these private models."""
import time
import uuid
from sqlalchemy import Column, JSON, UniqueConstraint, Index, text
from sqlmodel import Field, SQLModel


def now() -> int:
    return int(time.time())


def identity() -> str:
    return uuid.uuid4().hex


class User(SQLModel, table=True):
    id: str = Field(default_factory=identity, primary_key=True)
    email: str = Field(unique=True, index=True)
    full_name: str
    hashed_password: str
    role: str = Field(index=True)
    verified: bool = False
    active: bool = True
    token_version: int = 0
    is_demo: bool = False
    created_at: int = Field(default_factory=now)


class ActionToken(SQLModel, table=True):
    id: str = Field(default_factory=identity, primary_key=True)
    digest: str = Field(unique=True, index=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    purpose: str
    target_email: str = ""
    expires_at: int
    consumed_at: int | None = None


class Candidate(SQLModel, table=True):
    user_id: str = Field(foreign_key="user.id", primary_key=True)
    headline: str = ""
    about: str = ""
    city: str = ""
    industry: str = "ИТ"
    phone: str = ""
    telegram: str = ""
    experience_years: int = 0
    declared_specialization: str = "backend"
    declared_grade: str = "Junior"
    stack: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    soft_skills: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    work_format: str = "remote"
    employment: str = "full_time"
    salary_min: int = 0
    salary_max: int = 0
    published: bool = False
    processing_consent_at: int = 0
    publication_consent_at: int = 0
    consent_version: str = "2026-10-v1"
    specialization: str | None = Field(default=None, index=True)
    grade: str | None = Field(default=None, index=True)
    grade_changed_at: int = 0
    skills: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    skill_portfolio: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False, server_default="{}"))
    fsp_id: str = ""
    achievements: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    updated_at: int = Field(default_factory=now)


class Company(SQLModel, table=True):
    user_id: str = Field(foreign_key="user.id", primary_key=True)
    name: str = ""
    description: str = ""
    industry: str = ""
    website: str = ""
    contact: str = ""


class Attempt(SQLModel, table=True):
    id: str = Field(default_factory=identity, primary_key=True)
    candidate_id: str = Field(foreign_key="user.id", index=True)
    specialization: str
    grade: str
    kind: str = "grade"
    rubric_version: str = "2026-10-v1"
    seed: int
    questions: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    answers: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    result: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: str = "in_progress"
    created_at: int = Field(default_factory=now)
    finished_at: int | None = None
    expires_at: int
    is_demo: bool = False


class Evidence(SQLModel, table=True):
    id: str = Field(default_factory=identity, primary_key=True)
    candidate_id: str = Field(foreign_key="user.id", index=True)
    attempt_id: str = Field(foreign_key="attempt.id", index=True)
    skill: str
    family: str
    score: float
    created_at: int = Field(default_factory=now)
    is_demo: bool = False


class GradeAward(SQLModel, table=True):
    id: str = Field(default_factory=identity, primary_key=True)
    candidate_id: str = Field(foreign_key="user.id", index=True)
    attempt_id: str = Field(foreign_key="attempt.id", unique=True)
    specialization: str
    grade: str
    created_at: int = Field(default_factory=now)


class Need(SQLModel, table=True):
    id: str = Field(default_factory=identity, primary_key=True)
    employer_id: str = Field(foreign_key="user.id", index=True)
    title: str
    description: str = ""
    specialization: str = ""
    grade: str = ""
    skills: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    work_format: str = ""
    salary_min: int = 0
    salary_max: int = 0
    fsp_only: bool = False
    saved_ids: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    ranking: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False, server_default="{}"))
    created_at: int = Field(default_factory=now)


class Job(SQLModel, table=True):
    id: str = Field(default_factory=identity, primary_key=True)
    employer_id: str = Field(foreign_key="user.id", index=True)
    title: str
    description: str
    specialization: str
    grade: str
    skills: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    work_format: str
    salary_min: int
    salary_max: int
    currency: str = "RUB"
    active: bool = True
    created_at: int = Field(default_factory=now)


class Invitation(SQLModel, table=True):
    __table_args__ = (Index("uq_active_invitation", "employer_id", "candidate_id", unique=True,
        sqlite_where=text("status IN ('sent', 'viewed')"), postgresql_where=text("status IN ('sent', 'viewed')")),)
    id: str = Field(default_factory=identity, primary_key=True)
    employer_id: str = Field(foreign_key="user.id", index=True)
    candidate_id: str = Field(foreign_key="user.id", index=True)
    job_id: str | None = Field(default=None, foreign_key="job.id")
    title: str
    description: str
    salary_min: int
    salary_max: int
    currency: str = "RUB"
    contact: str
    status: str = "sent"
    created_at: int = Field(default_factory=now)
    updated_at: int = Field(default_factory=now)
    expires_at: int
    contact_revoked: bool = False


class Application(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("job_id", "candidate_id"),)
    id: str = Field(default_factory=identity, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    candidate_id: str = Field(foreign_key="user.id", index=True)
    employer_id: str = Field(foreign_key="user.id", index=True)
    message: str = ""
    status: str = "sent"
    created_at: int = Field(default_factory=now)
    updated_at: int = Field(default_factory=now)


class AuditEvent(SQLModel, table=True):
    id: str = Field(default_factory=identity, primary_key=True)
    actor_id: str = Field(index=True)
    action: str
    entity_id: str
    created_at: int = Field(default_factory=now)


class LocalMail(SQLModel, table=True):
    """Only the local demo mail transport stores messages; production uses SMTP."""
    id: str = Field(default_factory=identity, primary_key=True)
    receipt_hash: str = Field(index=True)
    email: str
    subject: str
    body: str
    created_at: int = Field(default_factory=now)
