"""
conftest.py — session-level setup to create the `ciba_test` Postgres database
if it doesn't already exist.

The `ciba_test` database is separate from `ciba_db` so test runs can't
accidentally corrupt development data.
"""
import asyncio
import os

import asyncpg

ADMIN_DSN = os.environ.get(
    "ADMIN_DATABASE_URL",
    "postgresql://ciba:ciba_dev@localhost:5432/ciba_db",
)
TEST_DB_NAME = "ciba_test"


def pytest_configure(config):
    """Create ciba_test database synchronously before any tests run."""
    async def _create_test_db():
        try:
            conn = await asyncpg.connect(ADMIN_DSN)
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB_NAME
            )
            if not exists:
                await conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
                print(f"\n[conftest] Created test database: {TEST_DB_NAME}")
            else:
                print(f"\n[conftest] Test database already exists: {TEST_DB_NAME}")
            await conn.close()
        except Exception as e:
            print(f"\n[conftest] Warning: could not ensure test database: {e}")
            print("[conftest] Make sure Postgres is running (docker-compose up -d postgres)")

    asyncio.run(_create_test_db())
