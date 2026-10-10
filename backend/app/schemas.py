import re
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

Specialization = Literal["backend", "data"]
Grade = Literal["Junior", "Middle"]
WorkFormat = Literal["remote", "hybrid", "office"]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Register(Input):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(min_length=2, max_length=100)
    role: Literal["candidate", "employer"]
    processing_consent: bool


class Verify(Input):
    token: str = Field(min_length=20, max_length=200)


class EmailRequest(Input):
    email: EmailStr


class ResetPassword(Verify):
    password: str = Field(min_length=10, max_length=128)


class ChangeEmail(Input):
    email: EmailStr
    password: str


class CandidateUpdate(Input):
    full_name: str = Field(min_length=2, max_length=100)
    headline: str = Field(default="", max_length=120)
    about: str = Field(default="", max_length=2000)
    city: str = Field(default="", max_length=80)
    industry: str = Field(default="ИТ", max_length=100)
    phone: str = Field(default="", max_length=40)
    telegram: str = Field(default="", max_length=80)
    experience_years: int = Field(default=0, ge=0, le=60)
    declared_specialization: Specialization = "backend"
    declared_grade: Grade = "Junior"
    stack: list[str] = Field(default_factory=list, max_length=20)
    soft_skills: list[str] = Field(default_factory=list, max_length=12)
    work_format: WorkFormat = "remote"
    employment: Literal["full_time", "part_time", "project"] = "full_time"
    salary_min: int = Field(default=0, ge=0, le=10000000)
    salary_max: int = Field(default=0, ge=0, le=10000000)
    published: bool = False
    processing_consent: bool = True
    publication_consent: bool = False

    @model_validator(mode="after")
    def consistency(self):
        if self.salary_max < self.salary_min:
            raise ValueError("Верхняя граница зарплаты должна быть не меньше нижней.")
        if self.published and (not self.processing_consent or not self.publication_consent):
            raise ValueError("Для публикации нужно согласие на обработку и публикацию профиля.")
        public = " ".join([self.headline, self.about, self.city, *self.stack, *self.soft_skills])
        if re.search(r"\S+@\S+|https?://|www\.|t\.me/|(?:\+?\d[\s()\-]*){7,}", public, re.I):
            raise ValueError("Контакты и ссылки указывайте в закрытых полях телефона и мессенджера.")
        if any(len(x) > 60 for x in self.stack + self.soft_skills):
            raise ValueError("Название навыка слишком длинное.")
        return self


class CompanyUpdate(Input):
    name: str = Field(min_length=2, max_length=120)
    description: str = Field(default="", max_length=3000)
    industry: str = Field(default="", max_length=120)
    website: str = Field(default="", max_length=250)
    contact: str = Field(min_length=3, max_length=200)

    @field_validator("website")
    @classmethod
    def website_scheme(cls, v):
        if v and not v.startswith(("https://", "http://")):
            raise ValueError("Адрес сайта должен начинаться с https:// или http://")
        return v


class AttemptStart(Input):
    specialization: Specialization
    grade: Grade
    kind: Literal["grade", "challenge"] = "grade"


class AnswerInput(Input):
    question_id: str = Field(max_length=80)
    choice: Annotated[int, Field(ge=0, le=10, strict=True)] | None = None
    values: dict[Annotated[str, Field(max_length=50)], Annotated[str, Field(max_length=40, strict=True)]] | None = Field(default=None, min_length=1, max_length=4)

    @model_validator(mode="after")
    def one_answer_type(self):
        if (self.choice is None) == (self.values is None):
            raise ValueError("Передайте один вариант ответа или числовые поля задания.")
        return self


PreferenceScore = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False, strict=True)]


class PreferenceAnswer(Input):
    left: dict[str, PreferenceScore] = Field(min_length=1, max_length=3)
    right: dict[str, PreferenceScore] = Field(min_length=1, max_length=3)
    choice: Literal["left", "right", "skip"]


class RankingConfig(Input):
    mode: Literal["evidence", "robust", "preference", "weighted", "mixed"] = "evidence"
    priorities: list[tuple[str, str]] = Field(default_factory=list, max_length=10)
    comparisons: list[PreferenceAnswer] = Field(default_factory=list, max_length=3)
    weights: dict[str, PreferenceScore] = Field(default_factory=dict, max_length=3, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def separate_preferences(self):
        if self.priorities and self.mode != "robust":
            raise ValueError("Порядок навыков относится к устойчивому режиму.")
        if self.comparisons and self.mode not in ("preference", "mixed"):
            raise ValueError("Ответы на уточнения относятся к подбору с уточнениями.")
        if self.weights and self.mode != "weighted":
            raise ValueError("Явные веса относятся к подбору по вашим приоритетам.")
        if self.weights and sum(self.weights.values()) <= 0:
            raise ValueError("Хотя бы один приоритет должен быть больше нуля.")
        return self


class SearchInput(Input):
    query: str = Field(default="", max_length=500)
    specialization: Literal["", "backend", "data"] = ""
    grade: Literal["", "Junior", "Middle"] = ""
    work_format: Literal["", "remote", "hybrid", "office"] = ""
    skills: list[str] = Field(default_factory=list, max_length=20)
    salary_min: int = Field(default=0, ge=0, le=10000000)
    salary_max: int = Field(default=0, ge=0, le=10000000)
    fsp_only: bool = False
    title: str = Field(default="Моя подборка", max_length=120)
    ranking: RankingConfig = Field(default_factory=RankingConfig)

    @model_validator(mode="after")
    def salary_range(self):
        if self.salary_max and self.salary_max < self.salary_min:
            raise ValueError("Проверьте диапазон зарплаты.")
        return self


class Offer(Input):
    title: str = Field(min_length=3, max_length=150)
    description: str = Field(min_length=10, max_length=5000)
    salary_min: int = Field(gt=0, le=10000000)
    salary_max: int = Field(gt=0, le=10000000)
    currency: Literal["RUB"] = "RUB"

    @model_validator(mode="after")
    def salary_range(self):
        if self.salary_max < self.salary_min:
            raise ValueError("Верхняя граница зарплаты должна быть не меньше нижней.")
        return self


class InviteInput(Offer):
    candidate_id: str
    contact: str = Field(min_length=3, max_length=200)
    job_id: str | None = None


class JobInput(Offer):
    specialization: Specialization
    grade: Grade
    skills: list[str] = Field(default_factory=list, max_length=20)
    work_format: WorkFormat = "remote"
    active: bool = True


class InviteAction(Input):
    action: Literal["view", "accept", "decline", "withdraw", "revoke_contacts"]


class ApplicationInput(Input):
    message: str = Field(default="", max_length=2000)
    contact_consent: bool


class ApplicationAction(Input):
    status: Literal["viewed", "interview", "declined", "withdrawn"]


class FspLink(Input):
    fsp_id: str = Field(default="", max_length=100)


class PublicUser(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    verified: bool
    is_demo: bool


class Contacts(BaseModel):
    full_name: str
    email: str
    phone: str
    telegram: str


class CandidateView(BaseModel):
    id: str
    name: str
    headline: str
    about: str
    city: str
    grade: str | None
    specialization: str | None
    work_format: str
    employment: str
    salary_min: int
    salary_max: int
    experience_years: int
    stack: list[str]
    skills: dict
    achievements: list
    evidence: list
    contacts: Contacts | None = None
    match_score: float = 0
    reasons: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    is_demo: bool
    category: str


class PriorityQuestion(BaseModel):
    skills: tuple[str, str]
    worst_after_points: float
    exact_choice: bool


class RobustReport(BaseModel):
    status: Literal["optimal", "bounded", "fallback"]
    considered: int
    selected_count: int
    worst_loss_points: float
    baseline_worst_loss_points: float
    nominal_loss_points: float
    optimization_gap_points: float
    worst_weights: dict[str, float]
    challenger_ids: list[str]
    question: PriorityQuestion | None = None
    model: str = "Наблюдаемые результаты; неизвестные приоритеты навыков. Не оценка вероятности найма."


class PreferenceQuestion(BaseModel):
    left: dict[str, float]
    right: dict[str, float]
    expected_gain_points: float


class PreferenceReport(BaseModel):
    engine: str = "linear-evoi-v3"
    status: Literal["question", "complete", "not_needed", "empty"]
    considered: int
    selected_count: int
    answered: int
    skipped: int
    remaining: int
    weights: dict[str, float]
    changed_count: int
    question: PreferenceQuestion | None = None
    model: str = "Приоритеты по ответам работодателя; баллы проверок не являются вероятностью успешного найма."


class WeightedReport(BaseModel):
    considered: int
    selected_count: int
    weights: dict[str, float]
    changed_count: int
    model: str = "Приоритеты заданы работодателем; веса не меняют обязательные условия, грейд или доступ к контактам."


class SearchResult(BaseModel):
    candidates: list[CandidateView]
    total: int
    normalized: SearchInput
    notices: list[str]
    algorithm: str = "evidence-v1"
    robust: RobustReport | None = None
    preference: PreferenceReport | None = None
    weighted: WeightedReport | None = None
