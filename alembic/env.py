"""
Alembic env.py — async-compatible, reads DATABASE_URL from app.config.

Alembic doesn't support asyncpg natively in run_migrations_online, so we use
the standard "run sync inside asyncio.run" pattern recommended in the Alembic
async docs (using `connectable.sync_engine` for the connection).
"""
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# ---- pull in the ORM metadata so autogenerate can diff ----
from app.models import Base  # noqa: F401  (side-effect: registers all models)
from app.config import settings

# Alembic Config object (gives access to alembic.ini values)
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata used by autogenerate
target_metadata = Base.metadata

# Override the placeholder sqlalchemy.url from alembic.ini with the real URL
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection (generates SQL script).
    Uses the URL string directly.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations via its sync counterpart."""
    connectable = create_async_engine(settings.DATABASE_URL, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
