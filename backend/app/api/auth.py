from app.responses import DemoMail, Login, MailSent, Message
import secrets
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import update
from sqlmodel import select
from app.core.config import settings
from app.core.security import Current, Db, digest, hash_password, issue_access, verify_password
from app.core.mail import send_mail
from app.core.rate_limit import limit
from app.models import ActionToken, AuditEvent, Candidate, Company, LocalMail, User, now
from app.schemas import ChangeEmail, EmailRequest, PublicUser, Register, ResetPassword, Verify

router = APIRouter(prefix="/auth", tags=["Регистрация и доступ"])


def user_data(user: User) -> dict:
    return PublicUser.model_validate(user, from_attributes=True).model_dump()


def send_action(db, user, purpose, email=None):
    value = secrets.token_urlsafe(32)
    address = email or user.email
    token = ActionToken(digest=digest(value), user_id=user.id, purpose=purpose,
                        target_email=address, expires_at=now() + (86400 if purpose == "verify" else 1800))
    db.add(token)
    page = "reset-password" if purpose == "reset" else "verify"
    label = "Восстановление пароля" if purpose == "reset" else "Подтверждение почты"
    link = f"{settings.frontend_url}/{page}?token={value}"
    receipt = send_mail(db, address, f"ФСП Карьера — {label}",
                       f"{label}\n\nОткройте ссылку: {link}\n\nЕсли вы не запрашивали это письмо, проигнорируйте его.")
    return receipt


def consume(db, value: str, purpose: tuple[str, ...]):
    token = db.exec(select(ActionToken).where(ActionToken.digest == digest(value))).first()
    if not token or token.purpose not in purpose or token.consumed_at or token.expires_at <= now():
        raise HTTPException(400, "Ссылка недействительна или уже использована. Запросите новое письмо.")
    changed = db.exec(update(ActionToken).where(ActionToken.id == token.id,
                      ActionToken.consumed_at.is_(None), ActionToken.expires_at > now()).values(consumed_at=now()))
    if changed.rowcount != 1:
        raise HTTPException(409, "Ссылка уже использована.")
    return token, db.get(User, token.user_id)


@router.post("/register", status_code=201, response_model=MailSent)
def register(data: Register, db: Db, request: Request) -> dict:
    """Создать неподтверждённый аккаунт и отправить одноразовую ссылку."""
    limit(request, "register", 12)
    if not data.processing_consent:
        raise HTTPException(422, "Необходимо согласие на обработку данных.")
    email = str(data.email).lower()
    if db.exec(select(User).where(User.email == email)).first():
        raise HTTPException(409, "Такой адрес уже зарегистрирован. Войдите или восстановите пароль.")
    user = User(email=email, full_name=data.full_name, role=data.role, hashed_password=hash_password(data.password))
    db.add(user)
    db.flush()
    if user.role == "candidate":
        db.add(Candidate(user_id=user.id, processing_consent_at=now()))
    else:
        db.add(Company(user_id=user.id, contact=email))
    receipt = send_action(db, user, "verify")
    db.add(AuditEvent(actor_id=user.id, action="register", entity_id=user.id))
    db.commit()
    return {"message": "Подтвердите адрес по ссылке в письме.", "email": email, "local_mail_receipt": receipt}


@router.post("/verify", response_model=Message)
def verify(data: Verify, db: Db, request: Request) -> dict:
    """Подтвердить адрес; повторное использование и просрочка отклоняются."""
    limit(request, "verify", 30)
    token, user = consume(db, data.token, ("verify", "email_change"))
    if not user:
        raise HTTPException(400, "Аккаунт не найден.")
    if token.purpose == "email_change":
        other = db.exec(select(User).where(User.email == token.target_email, User.id != user.id)).first()
        if other:
            raise HTTPException(409, "Этот адрес уже занят.")
        user.email = token.target_email
        user.token_version += 1
    elif token.target_email != user.email:
        raise HTTPException(400, "Адрес изменился. Запросите новое подтверждение.")
    user.verified = True
    db.add(user)
    db.add(AuditEvent(actor_id=user.id, action="email_verified", entity_id=user.id))
    db.commit()
    return {"message": "Почта подтверждена. Теперь можно войти."}


@router.post("/resend", response_model=MailSent)
def resend(data: EmailRequest, db: Db, request: Request) -> dict:
    limit(request, "email", 8)
    user = db.exec(select(User).where(User.email == str(data.email).lower())).first()
    receipt = send_action(db, user, "verify") if user and not user.verified else None
    db.commit()
    return {"message": "Если адрес ожидает подтверждения, письмо отправлено.", "local_mail_receipt": receipt}


@router.post("/login", response_model=Login)
def login(db: Db, request: Request, form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> dict:
    limit(request, "login", 30)
    user = db.exec(select(User).where(User.email == form.username.lower().strip())).first()
    if not user or (user.is_demo and not settings.demo_mode) or not verify_password(form.password, user.hashed_password):
        raise HTTPException(401, "Неверный email или пароль.")
    if not user.active:
        raise HTTPException(403, "Аккаунт отключён.")
    if not user.verified:
        raise HTTPException(403, "Сначала подтвердите email по ссылке в письме.")
    return {"access_token": issue_access(user), "token_type": "bearer", "user": user_data(user)}


@router.post("/demo/{role}", response_model=Login)
def demo(role: str, db: Db, request: Request) -> dict:
    """Только демонстрационный стенд: вход в явно синтетическую учётную запись."""
    limit(request, "demo", 30)
    if not settings.demo_mode or role not in ("candidate", "employer", "employer2"):
        raise HTTPException(404, "Демонстрационный вход недоступен.")
    user = db.exec(select(User).where(User.email == f"{role}@demo.fsp.test", User.is_demo == True)).first()
    if not user:
        raise HTTPException(503, "Демонстрационные данные ещё не загружены.")
    return {"access_token": issue_access(user), "token_type": "bearer", "user": user_data(user)}


@router.get("/me", response_model=PublicUser)
def me(user: Current):
    return user_data(user)


@router.post("/logout", response_model=Message)
def logout(user: Current, db: Db) -> dict:
    user.token_version += 1
    db.add(user)
    db.commit()
    return {"message": "Сессии завершены."}


@router.post("/forgot-password", response_model=MailSent)
def forgot(data: EmailRequest, db: Db, request: Request) -> dict:
    limit(request, "email", 8)
    user = db.exec(select(User).where(User.email == str(data.email).lower())).first()
    receipt = send_action(db, user, "reset") if user else None
    db.commit()
    return {"message": "Если аккаунт существует, письмо для восстановления отправлено.", "local_mail_receipt": receipt}


@router.post("/reset-password", response_model=Message)
def reset(data: ResetPassword, db: Db, request: Request) -> dict:
    limit(request, "reset", 20)
    _, user = consume(db, data.token, ("reset",))
    if not user:
        raise HTTPException(400, "Аккаунт не найден.")
    user.hashed_password = hash_password(data.password)
    user.token_version += 1
    db.add(user)
    db.commit()
    return {"message": "Пароль изменён. Войдите с новым паролем."}


@router.post("/change-email", response_model=MailSent)
def change_email(data: ChangeEmail, user: Current, db: Db, request: Request) -> dict:
    limit(request, "email", 8)
    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(403, "Проверьте текущий пароль.")
    email = str(data.email).lower()
    if db.exec(select(User).where(User.email == email)).first():
        raise HTTPException(409, "Адрес уже используется.")
    # Current verified address stays active until the new address is proved.
    receipt = send_action(db, user, "email_change", email)
    db.commit()
    return {"message": "Подтвердите новый адрес по ссылке в письме.", "local_mail_receipt": receipt}


@router.get("/local-mail/{receipt}", response_model=DemoMail)
def local_mail(receipt: str, db: Db) -> dict:
    """Локальная почта: случайный секрет квитанции, только DEMO_MODE, срок сутки."""
    if not settings.demo_mode:
        raise HTTPException(404)
    mail = db.exec(select(LocalMail).where(LocalMail.receipt_hash == digest(receipt),
                   LocalMail.created_at > now() - 86400)).first()
    if not mail:
        raise HTTPException(404, "Письмо не найдено или устарело.")
    return {"email": mail.email, "subject": mail.subject, "body": mail.body,
            "transport": "local_demo", "notice": "Это локальная демонстрация почты, письмо не отправлено во внешний ящик."}
