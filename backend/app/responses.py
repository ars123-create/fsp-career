"""Explicit response contracts for OpenAPI; no persistence model is serialized blindly."""
from typing import Literal
from pydantic import BaseModel, Field
from app.schemas import PublicUser, CandidateView, SearchInput, SearchResult


class ValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ApiError(BaseModel):
    detail: str | list[ValidationIssue]


class Message(BaseModel):
    message: str


class MailSent(Message):
    local_mail_receipt: str | None = None
    email: str | None = None


class Login(BaseModel):
    access_token: str
    token_type: Literal["bearer"]
    user: PublicUser


class DemoMail(BaseModel):
    email: str
    subject: str
    body: str
    transport: Literal["local_demo"]
    notice: str


class Skill(BaseModel):
    score: float
    families: int
    checked_at: int
    source: str
    latest_checked_at: int | None = Field(default=None, exclude_if=lambda value: value is None)
    rubric_version: str | None = Field(default=None, exclude_if=lambda value: value is None)


class OwnProfile(BaseModel):
    user_id: str
    full_name: str
    email: str
    headline: str
    about: str
    city: str
    industry: str
    phone: str
    telegram: str
    experience_years: int
    declared_specialization: str
    declared_grade: str
    stack: list[str]
    soft_skills: list[str]
    work_format: str
    employment: str
    salary_min: int
    salary_max: int
    published: bool
    processing_consent: bool
    publication_consent: bool
    processing_consent_at: int
    publication_consent_at: int
    consent_version: str
    specialization: str | None
    grade: str | None
    grade_changed_at: int
    next_grade_change_at: int
    skills: dict[str, Skill]
    fsp_id: str
    fsp_status: str
    achievements: list[dict]
    updated_at: int


class CompanyProfile(BaseModel):
    name: str
    description: str
    industry: str
    website: str
    contact: str


class CompanyOwn(CompanyProfile):
    user_id: str


class Rubric(BaseModel):
    version: str
    pass_score: int
    min_skill_score: int
    min_practical_skill_score: int = 0
    notice: str


class AnswerField(BaseModel):
    name: str
    label: str
    places: int


class QuestionTable(BaseModel):
    name: str
    columns: list[str]
    rows: list[list[str | int | float | None]]


class Question(BaseModel):
    id: str
    family: str
    skill: str
    title: str
    prompt: str
    code: str
    choices: list[str]
    answer_type: Literal["choice", "fields"] = "choice"
    fields: list[AnswerField] = Field(default_factory=list)
    tables: list[QuestionTable] = Field(default_factory=list)
    correct_choice: int | None = None
    correct_values: dict[str, str] | None = None
    is_correct: bool | None = None
    explanation: str | None = None


class Result(BaseModel):
    score: float = 0
    correct: int = 0
    total: int = 0
    skills: dict[str, float] = Field(default_factory=dict)
    practical_skills: dict[str, float] = Field(default_factory=dict)
    passed: bool = False
    grade_outcome: str | None = None
    evidence_outcome: str | None = None
    current_grade: str | None = None
    next_grade_change_at: int = 0
    rubric: Rubric | None = None


class Assessment(BaseModel):
    id: str
    specialization: str
    grade: str
    kind: str
    status: str
    questions: list[Question]
    answers: dict[str, int | dict[str, str]]
    result: Result
    created_at: int
    expires_at: int
    finished_at: int | None
    rubric: Rubric
    rubric_version: str
    is_demo: bool


class Catalog(BaseModel):
    specializations: dict[str, str]
    grades: list[str]
    skills: dict[str, list[str]]
    question_count: int
    practical_count: int
    duration_minutes: int
    challenge_count: int
    challenge_duration_minutes: int
    rubric: Rubric
    cooldown_days: int


class AnswerSaved(BaseModel):
    saved: bool
    answered: int


class FspStatus(Message):
    status: Literal["pending", "not_linked"]
    achievements: list[dict]


class ImportSuggestion(BaseModel):
    phone: str
    stack: list[str]


class PdfImport(Message):
    preview: str
    suggested: ImportSuggestion
    found_email: str


class NeedView(BaseModel):
    id: str
    title: str
    description: str
    specialization: str
    grade: str
    skills: list[str]
    work_format: str
    salary_min: int
    salary_max: int
    fsp_only: bool
    saved_ids: list[str]
    ranking: dict = {}
    created_at: int


class ParsedSearch(BaseModel):
    normalized: SearchInput
    notices: list[str]


class RestoredNeed(BaseModel):
    need: NeedView
    previous: list[CandidateView]
    current: SearchResult


class Deleted(BaseModel):
    deleted: bool


class OfferView(BaseModel):
    id: str
    employer_id: str
    title: str
    description: str
    salary_min: int
    salary_max: int
    currency: Literal["RUB"]
    created_at: int
    company: CompanyProfile


class InvitationView(OfferView):
    candidate_id: str
    job_id: str | None
    contact: str
    status: str
    updated_at: int
    expires_at: int
    contact_revoked: bool
    candidate: CandidateView | None


class JobView(OfferView):
    specialization: str
    grade: str
    skills: list[str]
    work_format: str
    active: bool
    application_id: str | None
    application_status: str | None


class ApplicationView(BaseModel):
    id: str
    job_id: str
    candidate_id: str
    employer_id: str
    message: str
    status: str
    created_at: int
    updated_at: int
    job: JobView | None = None
    candidate: CandidateView | None = None


class Dashboard(BaseModel):
    invitations: int
    accepted: int
    pending: int
    grade: str | None = None
    specialization: str | None = None
    published: bool | None = None
    skills: dict[str, Skill] | None = None
    attempts: int | None = None
    candidates: int | None = None
    jobs: int | None = None


class Health(BaseModel):
    status: str
    version: str
    demo_mode: bool
    mail_transport: str
