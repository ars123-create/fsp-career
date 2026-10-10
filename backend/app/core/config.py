from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FSP_", env_file=str(ROOT / ".env"), extra="ignore")
    secret_key: str
    database_url: str = f"sqlite:///{ROOT / 'data' / 'fsp.db'}"
    frontend_url: str = "http://127.0.0.1:5178"
    demo_mode: bool = False
    smtp_host: str = ""
    smtp_port: int = 1026
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_tls: bool = False
    email_from: str = "career@fsp.example"
    access_minutes: int = 480
    grade_cooldown_days: int = 90


settings = Settings()
if len(settings.secret_key) < 32:
    raise RuntimeError("FSP_SECRET_KEY must contain at least 32 characters")
