"""Adapted from Alembic's generic environment (MIT); URL stays in settings."""
from alembic import context
from sqlalchemy import create_engine, pool
from app.core.config import settings
from app.models import SQLModel

target_metadata = SQLModel.metadata
if context.is_offline_mode():
    context.configure(url=settings.database_url, target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction(): context.run_migrations()
else:
    engine = create_engine(settings.database_url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction(): context.run_migrations()
