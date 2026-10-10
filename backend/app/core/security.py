"""Adapted from fastapi/full-stack-fastapi-template, MIT (see licenses/).

Adds audience, purpose and DB token-version checks; hashing remains pwdlib Argon2.
"""
from datetime import UTC, datetime, timedelta
import hashlib
import secrets
from typing import Annotated
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from sqlmodel import Session
from app.core.config import settings
from app.core.db import get_db
from app.models import User

password_hash = PasswordHash((Argon2Hasher(),))
oauth = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
Db = Annotated[Session, Depends(get_db)]


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return password_hash.verify(password, hashed)
    except (ValueError, TypeError):
        return False


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def issue_access(user: User) -> str:
    stamp = datetime.now(UTC)
    return jwt.encode({"sub": user.id, "aud": "fsp-career", "purpose": "access",
                       "iat": stamp, "exp": stamp + timedelta(minutes=settings.access_minutes),
                       "ver": user.token_version, "jti": secrets.token_hex(12)},
                      settings.secret_key, algorithm="HS256")


def current_user(db: Db, token: Annotated[str, Depends(oauth)]) -> User:
    try:
        p = jwt.decode(token, settings.secret_key, algorithms=["HS256"], audience="fsp-career",
                       options={"require": ["exp", "iat", "sub", "aud", "purpose", "ver"]})
        user = db.get(User, p["sub"])
        if not user or (user.is_demo and not settings.demo_mode) or not user.active or not user.verified or p["purpose"] != "access" or p["ver"] != user.token_version:
            raise ValueError()
        return user
    except (jwt.InvalidTokenError, ValueError, KeyError):
        raise HTTPException(401, "Войдите снова: сессия недействительна или завершена.")


Current = Annotated[User, Depends(current_user)]


def candidate_user(user: Current) -> User:
    if user.role != "candidate":
        raise HTTPException(403, "Это действие доступно кандидату.")
    return user


def employer_user(user: Current) -> User:
    if user.role != "employer":
        raise HTTPException(403, "Это действие доступно работодателю.")
    return user


CandidateUser = Annotated[User, Depends(candidate_user)]
EmployerUser = Annotated[User, Depends(employer_user)]
