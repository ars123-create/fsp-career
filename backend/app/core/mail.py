"""Standard SMTP delivery; explicitly local receipt-protected fallback in demo mode."""
from email.message import EmailMessage
import secrets
import smtplib
from fastapi import HTTPException
from sqlmodel import Session
from app.core.config import settings
from app.core.security import digest
from app.models import LocalMail


def send_mail(db: Session, email: str, subject: str, body: str) -> str | None:
    if settings.smtp_host:
        message = EmailMessage()
        message["From"] = settings.email_from
        message["To"] = email
        message["Subject"] = subject
        message.set_content(body)
        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
                if settings.smtp_tls:
                    server.starttls()
                if settings.smtp_user:
                    server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(message)
        except (OSError, smtplib.SMTPException):
            raise HTTPException(503, "Почтовый сервис временно недоступен. Повторите запрос позже.")
        return None
    if not settings.demo_mode:
        raise HTTPException(503, "Доставка почты не настроена.")
    receipt = secrets.token_urlsafe(32)
    db.add(LocalMail(receipt_hash=digest(receipt), email=email, subject=subject, body=body))
    return receipt
