from collections.abc import Generator
from sqlalchemy import event
from sqlmodel import Session, create_engine
from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True,
    connect_args={"check_same_thread": False, "timeout": 20} if settings.database_url.startswith("sqlite") else {})

if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def sqlite_setup(connection, _):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def get_db() -> Generator[Session]:
    with Session(engine) as session:
        yield session
